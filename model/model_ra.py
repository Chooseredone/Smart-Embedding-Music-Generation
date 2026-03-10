"""
model_ra.py - Music Transformer with Smart Embedding
Enhanced with pitch/hand decomposition for better generalization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict
import logging
from torch.utils.checkpoint import checkpoint

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Flash Attention 2 support (optional)
try:
    from flash_attn import flash_attn_func
    FLASH_ATTENTION_AVAILABLE = True
    logger.info("Flash Attention 2 is available")
except ImportError:
    FLASH_ATTENTION_AVAILABLE = False
    logger.info("Flash Attention 2 not available, using standard attention")


class LayerNorm(nn.Module):
    """LayerNorm with optional bias"""
    def __init__(self, ndim, bias=False):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    """Multi-head self attention with optional RoPE/ALiBi"""
    
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        # Key, query, value projections for all heads
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.d_k = config.n_embd // config.n_head
        self.dropout = config.dropout
        self.use_ra = getattr(config, 'use_ra', True)
        self.use_alibi = getattr(config, 'use_alibi', False)
        self.rope_base = getattr(config, 'rope_base', 1000)
        
        # Validate rope_base
        if self.use_ra and not self.use_alibi:
            if self.rope_base < 100:
                logger.warning(f"Low rope_base ({self.rope_base}) may cause overfitting on short sequences")
            elif self.rope_base > 100000:
                logger.warning(f"High rope_base ({self.rope_base}) may reduce position sensitivity")
        
        # QKV projection
        self.qkv_proj = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        
        # Output projection
        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        
        # Regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        # Flash attention flag
        self.flash = hasattr(config, 'use_flash_attn') and config.use_flash_attn and FLASH_ATTENTION_AVAILABLE
        if not self.flash:
            # Causal mask for standard attention
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))
        
        # ALiBi slopes if enabled
        if self.use_alibi:
            slopes = self._get_alibi_slopes(self.n_head)
            self.register_buffer('alibi_slopes', slopes)
    
    def _get_alibi_slopes(self, n_heads: int) -> torch.Tensor:
        """Get ALiBi slopes for each attention head"""
        def get_slopes_power_of_2(n):
            start = (2**(-2**-(math.log2(n)-3)))
            ratio = start
            return [start*ratio**i for i in range(n)]
        
        if math.log2(n_heads).is_integer():
            slopes = torch.Tensor(get_slopes_power_of_2(n_heads))
        else:
            closest_power_of_2 = 2**math.floor(math.log2(n_heads))
            logger.debug(f"ALiBi: {n_heads} heads → using closest power {closest_power_of_2}")
            slopes = torch.Tensor(get_slopes_power_of_2(closest_power_of_2))
            slopes = torch.cat([slopes, slopes[::2][:n_heads-closest_power_of_2]])
        
        return slopes.view(1, n_heads, 1, 1)
    
    def _apply_rotary_emb(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Rotary Position Embeddings to q and k"""
        # x shape: [B, n_head, T, d_k]
        B, H, T, D = x.shape
        
        # Position indices: [T]
        positions = torch.arange(T, device=x.device, dtype=torch.float)
        
        # Frequencies with explicit dim//2
        inv_freq = 1.0 / (self.rope_base ** (torch.arange(0, D//2, 1, device=x.device, dtype=torch.float) / (D//2)))
        
        # Compute angles: [T, D//2]
        angles = positions.unsqueeze(-1) * inv_freq.unsqueeze(0)
        
        # Sin and cos: [T, D//2]
        sin = torch.sin(angles)
        cos = torch.cos(angles)
        
        # Expand for batch and heads: [1, 1, T, D//2]
        sin = sin.unsqueeze(0).unsqueeze(0)
        cos = cos.unsqueeze(0).unsqueeze(0)
        
        # Reshape x for rotation
        x_reshaped = x.reshape(B, H, T, D//2, 2)
        
        # Apply rotation
        x_rot = torch.empty_like(x_reshaped)
        x_rot[..., 0] = x_reshaped[..., 0] * cos - x_reshaped[..., 1] * sin
        x_rot[..., 1] = x_reshaped[..., 0] * sin + x_reshaped[..., 1] * cos
        
        # Reshape back
        return x_rot.reshape(B, H, T, D)
    
    def _apply_alibi(self, scores: torch.Tensor, T: int) -> torch.Tensor:
        """Apply ALiBi (Attention with Linear Biases)"""
        # Create position bias matrix
        position_bias = torch.abs(torch.arange(T, device=scores.device).unsqueeze(0) - 
                                 torch.arange(T, device=scores.device).unsqueeze(1))
        position_bias = position_bias.unsqueeze(0).unsqueeze(0)  # [1, 1, T, T]
        
        # Apply slopes
        alibi_bias = -(position_bias * self.alibi_slopes)
        
        return scores + alibi_bias
    
    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, C = x.size()
        
        # Log long sequences
        if T > self.n_embd:
            logger.debug(f"Processing long sequence: T={T}, using {'RoPE' if self.use_ra else 'ALiBi' if self.use_alibi else 'absolute'} position encoding")
        
        # QKV projection and split
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        
        # Reshape for multi-head attention
        q = q.view(B, T, self.n_head, self.d_k).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.d_k).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_k).transpose(1, 2)
        
        # Apply RoPE if enabled (not with ALiBi)
        if self.use_ra and not self.use_alibi:
            q = self._apply_rotary_emb(q)
            k = self._apply_rotary_emb(k)
        
        # Attention computation
        if self.flash:
            q = q.transpose(1, 2).contiguous()
            k = k.transpose(1, 2).contiguous()
            v = v.transpose(1, 2).contiguous()
            
            y = flash_attn_func(q, k, v, dropout_p=self.dropout if self.training else 0, causal=True)
            y = y.view(B, T, C)
        else:
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
            
            # Apply ALiBi if enabled
            if self.use_alibi:
                scores = self._apply_alibi(scores, T)
            
            # Apply causal mask
            scores = scores.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            
            if attn_mask is not None:
                scores = scores.masked_fill(attn_mask == 0, float('-inf'))
            
            att = F.softmax(scores, dim=-1)
            att = self.attn_dropout(att)
            
            y = torch.matmul(att, v)
            y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        y = self.resid_dropout(self.out_proj(y))
        return y


class MLP(nn.Module):
    """MLP block with GELU activation"""
    
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer block with style conditioning via AdaLN"""
    
    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)
        
        self.ln1 = LayerNorm(config.n_embd, bias=config.bias)
        self.ln2 = LayerNorm(config.n_embd, bias=config.bias)
        
        self.style_proj1 = nn.Linear(config.n_embd, 2 * config.n_embd, bias=True)
        self.style_proj2 = nn.Linear(config.n_embd, 2 * config.n_embd, bias=True)
        
        self.use_checkpoint = getattr(config, 'gradient_checkpointing', False)
    
    def _forward(self, x: torch.Tensor, style_emb: torch.Tensor) -> torch.Tensor:
        """Forward pass implementation"""
        # AdaLN for attention
        shift1, scale1 = self.style_proj1(style_emb).chunk(2, dim=-1)
        shift1 = shift1.unsqueeze(1)
        scale1 = scale1.unsqueeze(1)
        
        x = x + self.attn(self.ln1(x) * (1 + scale1) + shift1)
        
        # AdaLN for MLP
        shift2, scale2 = self.style_proj2(style_emb).chunk(2, dim=-1)
        shift2 = shift2.unsqueeze(1)
        scale2 = scale2.unsqueeze(1)
        
        x = x + self.mlp(self.ln2(x) * (1 + scale2) + shift2)
        
        return x
    
    def forward(self, x: torch.Tensor, style_emb: torch.Tensor) -> torch.Tensor:
        if self.use_checkpoint and self.training:
            return checkpoint(self._forward, x, style_emb)
        else:
            return self._forward(x, style_emb)


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class SmartMusicEmbedding(nn.Module):
    """Smart embedding with safety checks"""
    
    def __init__(self, vocab_size: int, n_embd: int, use_smart_embedding: bool = True):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.use_smart_embedding = use_smart_embedding
        self._mappings_initialized = False  # Added!
        
        # Standard token embedding
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        
        if use_smart_embedding:
            self.pitch_embedding = nn.Embedding(88, n_embd)
            self.hand_embedding = nn.Embedding(3, n_embd)
            
            self.register_buffer('pitch_map', torch.zeros(vocab_size, dtype=torch.long))
            self.register_buffer('hand_map', torch.zeros(vocab_size, dtype=torch.long))
            self.register_buffer('is_note_token', torch.zeros(vocab_size, dtype=torch.bool))
            
            logger.info("Smart Embedding initialized (mappings pending)")
    
    def setup_mappings(self, vocab: Dict[str, int]):
        """Setup mappings from vocabulary"""
        if not self.use_smart_embedding:
            self._mappings_initialized = True
            return
        
        import re
        note_on_pattern = re.compile(r'(L_HAND|R_HAND|LH|RH)_NOTE_ON_(\d+)')
        note_off_pattern = re.compile(r'(L_HAND|R_HAND|LH|RH)_NOTE_OFF_(\d+)')
        
        notes_found = 0
        for token, token_id in vocab.items():
            match = note_on_pattern.match(token) or note_off_pattern.match(token)
            
            if match:
                hand_str = match.group(1)
                pitch = int(match.group(2))
                
                if hand_str in ['RH', 'R_HAND']:
                    hand = 1
                elif hand_str in ['LH', 'L_HAND']:
                    hand = 2
                else:
                    hand = 0
                
                if 21 <= pitch <= 108:
                    self.pitch_map[token_id] = pitch - 21
                    self.hand_map[token_id] = hand
                    self.is_note_token[token_id] = True
                    notes_found += 1
        
        if notes_found == 0:
            raise ValueError(
                "No note tokens found in vocabulary! "
                "Check if vocab contains RH_NOTE_ON_*, LH_NOTE_ON_* tokens"
            )
        
        self._mappings_initialized = True  # Success!
        logger.info(f"✅ Smart Embedding mappings created: {notes_found} note tokens")
    
    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        # Safety check: Check on the first forward pass
        if self.use_smart_embedding and not self._mappings_initialized:
            raise RuntimeError(
                "Smart Embedding mappings not initialized! "
                "Call model.setup_vocab_mappings(vocab) after creating the model."
            )
        
        if not self.use_smart_embedding:
            return self.token_embedding(idx)
        
        base_emb = self.token_embedding(idx)
        note_mask = self.is_note_token[idx]
        
        if note_mask.any():
            note_indices = idx[note_mask]
            pitch_ids = self.pitch_map[note_indices]
            hand_ids = self.hand_map[note_indices]
            
            pitch_emb = self.pitch_embedding(pitch_ids)
            hand_emb = self.hand_embedding(hand_ids)
            smart_emb = pitch_emb + hand_emb
            
            base_emb[note_mask] = smart_emb
        
        return base_emb
    
    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """
        Apply smart embedding to input tokens
        Args:
            idx: [batch_size, seq_len] tensor of token ids
        Returns:
            [batch_size, seq_len, n_embd] tensor of embeddings
        """
        if not self.use_smart_embedding:
            return self.token_embedding(idx)
        
        # Get base embeddings for all tokens
        base_emb = self.token_embedding(idx)
        
        # Find note tokens
        note_mask = self.is_note_token[idx]
        
        if note_mask.any():
            # Get pitch and hand for note tokens
            note_indices = idx[note_mask]
            pitch_ids = self.pitch_map[note_indices]
            hand_ids = self.hand_map[note_indices]
            
            # Get pitch and hand embeddings
            pitch_emb = self.pitch_embedding(pitch_ids)
            hand_emb = self.hand_embedding(hand_ids)
            
            # Combine pitch and hand embeddings
            smart_emb = pitch_emb + hand_emb
            
            # Replace base embeddings with smart embeddings for note tokens
            base_emb[note_mask] = smart_emb
        
        return base_emb


class ConditionalMusicTransformer(nn.Module):
    """Music Transformer with style conditioning and Smart Embedding"""
    
    def __init__(self, vocab_size: int, n_styles: int = 75, block_size: int = 2048,
                 n_layer: int = 12, n_head: int = 12, n_embd: int = 768,
                 dropout: float = 0.1, bias: bool = False, use_ra: bool = True,
                 use_alibi: bool = False, rope_base: int = 1000, 
                 gradient_checkpointing: bool = False,
                 pad_token_id: int = 0, eos_token_id: Optional[int] = None,
                 use_smart_embedding: bool = True):
        super().__init__()
        
        # Validate position encoding choice
        if use_ra and use_alibi:
            logger.warning("Both RoPE and ALiBi enabled, using ALiBi only")
            use_ra = False
        
        # Store all config parameters
        self.config = type('Config', (), {
            'vocab_size': vocab_size,
            'n_styles': n_styles,
            'block_size': block_size,
            'n_layer': n_layer,
            'n_head': n_head,
            'n_embd': n_embd,
            'dropout': dropout,
            'bias': bias,
            'use_flash_attn': FLASH_ATTENTION_AVAILABLE,
            'use_ra': use_ra,
            'use_alibi': use_alibi,
            'rope_base': rope_base,
            'gradient_checkpointing': gradient_checkpointing,
            'pad_token_id': pad_token_id,
            'eos_token_id': eos_token_id,
            'use_smart_embedding': use_smart_embedding
        })()
        
        # Smart embedding module
        self.embedding = SmartMusicEmbedding(vocab_size, n_embd, use_smart_embedding)
        
        # Style embedding
        self.style_embedding = nn.Embedding(n_styles, n_embd)
        
        # Positional encoding (only if not using RoPE/ALiBi)
        self.use_ra = use_ra
        self.use_alibi = use_alibi
        if not use_ra and not use_alibi:
            self.pos_encoding = SinusoidalPositionalEncoding(n_embd, block_size)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([TransformerBlock(self.config) for _ in range(n_layer)])
        
        # Final layer norm and output projection
        self.ln_f = LayerNorm(n_embd, bias=bias)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        
        # Weight tying between token embedding and lm_head
        self.embedding.token_embedding.weight = self.lm_head.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('out_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * n_layer))
        
        # Log configuration
        logger.info(f"Model initialized with position encoding: {'RoPE' if use_ra else 'ALiBi' if use_alibi else 'Absolute'}")
        logger.info(f"Smart Embedding: {'Enabled' if use_smart_embedding else 'Disabled'}")
    
    def setup_vocab_mappings(self, vocab: Dict[str, int]):
        """Setup vocabulary mappings for smart embedding"""
        self.embedding.setup_mappings(vocab)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx: torch.Tensor, style_ids: torch.Tensor, 
                targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Sequence length {t} exceeds block size {self.config.block_size}"
        
        # Added warning for long sequences
        if t > self.config.block_size * 0.75:
            logger.warning(f"Sequence length {t} approaching limit {self.config.block_size}")
        
        # Smart token embeddings
        tok_emb = self.embedding(idx)
        
        # Position handling based on encoding type
        if self.use_ra or self.use_alibi:
            x = tok_emb  # RoPE/ALiBi handle positions
        else:
            x = self.pos_encoding(tok_emb)
        
        # Style embeddings
        style_emb = self.style_embedding(style_ids)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, style_emb)
        
        # Final layer norm and output
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        # Calculate loss if targets provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), 
                targets.view(-1), 
                ignore_index=self.config.pad_token_id
            )
        
        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx: torch.Tensor, style_id: int, max_new_tokens: int, 
                 temperature: float = 1.0, top_k: Optional[int] = None,
                 eos_token_id: Optional[int] = None) -> torch.Tensor:
        """Generate tokens autoregressively"""
        if eos_token_id is None:
            eos_token_id = self.config.eos_token_id
            
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            
            style_ids = torch.full((idx_cond.size(0),), style_id, device=idx.device)
            logits, _ = self(idx_cond, style_ids)
            logits = logits[:, -1, :] / temperature
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            idx = torch.cat((idx, idx_next), dim=1)
            
            # Early stopping on EOS token
            if eos_token_id is not None and (idx_next == eos_token_id).any():
                break
        
        return idx
    
    def get_num_params(self) -> int:
        """Get number of parameters"""
        return sum(p.numel() for p in self.parameters())
    
    def get_config_dict(self) -> Dict:
        """Get configuration dictionary for saving"""
        return {k: v for k, v in self.config.__dict__.items()}


def create_model(vocab_size: int, n_styles: int = 75, model_size: str = 'base', 
                 use_ra: bool = True, use_alibi: bool = False,
                 rope_base: int = 1000, gradient_checkpointing: bool = False,
                 pad_token_id: int = 0, eos_token_id: Optional[int] = None,
                 use_smart_embedding: bool = True) -> ConditionalMusicTransformer:
    """
    Create a Conditional Music Transformer model with Smart Embedding
    
    Args:
        vocab_size: Size of the vocabulary
        n_styles: Number of style classes  
        model_size: Model size ('small', 'base', 'large')
        use_ra: Whether to use Relative Attention (RoPE)
        use_alibi: Whether to use ALiBi (alternative to RoPE)
        rope_base: Base for RoPE frequencies (1000 for music, 10000 for text)
        gradient_checkpointing: Whether to use gradient checkpointing
        pad_token_id: Padding token id for loss calculation
        eos_token_id: End of sequence token id for generation
        use_smart_embedding: Whether to use smart embedding (pitch + hand decomposition)
    
    Returns:
        ConditionalMusicTransformer model
    """
    configs = {
        'small': dict(n_layer=6, n_head=8, n_embd=512),
        'base': dict(n_layer=12, n_head=12, n_embd=768),
        'large': dict(n_layer=24, n_head=16, n_embd=1024)
    }
    
    config = configs[model_size]
    
    return ConditionalMusicTransformer(
        vocab_size=vocab_size,
        n_styles=n_styles,
        use_ra=use_ra,
        use_alibi=use_alibi,
        rope_base=rope_base,
        gradient_checkpointing=gradient_checkpointing,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        use_smart_embedding=use_smart_embedding,
        **config
    )