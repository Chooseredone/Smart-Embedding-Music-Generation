"""
train_ra.py - Training script for Conditional Music Transformer with RoPE

This is a revised version aligned with train_v2_fixed.py, ensuring identical training
setup except for the model (RoPE vs. absolute attention). Includes transition token
support, fine-tuning, and consistent metrics.

Key Changes:
- Fixed vocab size handling for non-consecutive vocabulary
- Added safety checks for augmentation
- Safe default values for augmentation probabilities
- IMPROVED: Better initialization for new tokens during fine-tuning
- CRITICAL FIX: Always unfreeze embeddings/lm_head for new token learning
- ADDED: BF16 support for FlashAttention compatibility
"""

import os
import sys
import argparse
import random
import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Set
from datetime import datetime
from collections import Counter, defaultdict
import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import wandb

try:
    from transformers import get_cosine_schedule_with_warmup
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("transformers not available, using manual warmup")

from model_ra import ConditionalMusicTransformer, create_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class MusicDataset(Dataset):
    MIN_MIDI_PITCH = 21
    MAX_MIDI_PITCH = 108

    def __init__(
        self,
        sequences: torch.Tensor,
        vocab: Dict[str, int],
        inv_vocab: Dict[int, str],
        metadata: List[Dict],
        indices: Optional[List[int]] = None,
        augmentation_config: Optional[Dict] = None,
        transition_token_ids: Optional[Set[int]] = None,
        seed: int = 42,
        max_seq_length: int = 2048
    ):
        self.vocab = vocab
        self.inv_vocab = inv_vocab
        self.metadata = metadata
        self.transition_token_ids = transition_token_ids or set()
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.max_seq_length = max_seq_length
        
        # Store max vocab ID for safe clamping
        self.max_vocab_id = max(vocab.values())
        self.vocab_size = len(vocab)
        
        if indices is not None:
            self.sequences = sequences[indices]
            self.metadata = [metadata[i] for i in indices]
        else:
            self.sequences = sequences
            
        self.pad_token_id = vocab['<PAD>']
        self.eos_token_id = vocab['<EOS>']
        self.pos_unknown_token_id = vocab.get('<POS_UNKNOWN>', vocab.get('<UNK>', 0))
        
        # Define patterns BEFORE calling _extract_style_ids()
        self.style_pattern = re.compile(r'<STYLE_(\d+)>')
        self.note_on_pattern = re.compile(r'(L_HAND|R_HAND|LH|RH)_NOTE_ON_(\d+)')
        self.note_off_pattern = re.compile(r'(L_HAND|R_HAND|LH|RH)_NOTE_OFF_(\d+)')
        self.velocity_pattern = re.compile(r'VELOCITY_(\d+)')
        self.time_shift_pattern = re.compile(r'TIME_SHIFT_(\d+)')
        
        # Now extract style_ids (after patterns are defined)
        self.style_ids = self._extract_style_ids()
        
        # Safe default: augmentation OFF by default
        self.augmentation_config = augmentation_config or {
            'p_augment': 0.0,  # Changed default to 0.0 for safety
            'p_octave_shift': 0.0,  # Octave shift OFF by default (most risky)
            'p_velocity_scale': 0.3,
            'p_time_warp': 0.3,
            'octave_range': (-1, 1),
            'velocity_scale_range': (0.8, 1.2),
            'time_warp_range': (0.9, 1.1)
        }
        
        logger.info(f"Dataset created with {len(self)} sequences")
        logger.info(f"Vocab size: {self.vocab_size}, Max vocab ID: {self.max_vocab_id}")
        logger.info(f"Max sequence length: {self.max_seq_length}")
        if self.transition_token_ids:
            logger.info(f"Tracking {len(self.transition_token_ids)} transition tokens")

    def _extract_style_ids(self) -> List[int]:
        style_ids = []
        for seq_idx in range(len(self.sequences)):
            seq = self.sequences[seq_idx]
            style_id = None
            for token_id in seq:
                if token_id == self.pad_token_id or token_id == self.eos_token_id:
                    break
                token = self.inv_vocab.get(token_id.item(), '')
                match = self.style_pattern.match(token)
                if match:
                    style_id = int(match.group(1))
                    break
            if style_id is None:
                if seq_idx < 5:
                    logger.info(f"No style token found in sequence {seq_idx}, defaulting to 0")
                style_id = 0
            style_ids.append(style_id)
        return style_ids

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        seq = self.sequences[idx].clone()
        style_id = self.style_ids[idx]
        
        # Sequence length limit
        original_len = len(seq)
        if original_len > self.max_seq_length:
            # Log only first few truncations to avoid spam
            if hasattr(self, '_truncation_warned'):
                self._truncation_warned += 1
            else:
                self._truncation_warned = 1
            
            if self._truncation_warned <= 5:
                logger.debug(f"Truncating sequence {idx}: {original_len} -> {self.max_seq_length}")
            elif self._truncation_warned == 6:
                logger.debug("Further truncation warnings suppressed...")
            
            seq = seq[:self.max_seq_length]
        
        # Apply augmentation if configured
        if self.augmentation_config and self.rng.random() < self.augmentation_config.get('p_augment', 0.0):
            seq_len = (seq != self.pad_token_id).sum().item()
            if seq_len > 0:
                seq[:seq_len] = self._augment_sequence(seq[:seq_len])
        
        # Safety check: ensure all tokens are within valid range
        if (seq > self.max_vocab_id).any() or (seq < 0).any():
            logger.warning(f"Invalid token IDs in sequence {idx}: max {seq.max().item()}, min {seq.min().item()}")
            seq = torch.clamp(seq, min=0, max=self.max_vocab_id)
        
        x = seq[:-1]
        y = seq[1:]
        return x, y, style_id

    def set_epoch(self, epoch: int):
        """Set the random seed for this epoch"""
        self.rng = np.random.RandomState(self.seed + epoch)

    def _augment_sequence(self, seq: torch.Tensor) -> torch.Tensor:
        """Apply augmentation to a sequence"""
        seq_np = seq.numpy().copy()
        augmentations = []
        
        # Only add augmentations if their probability > 0
        if self.rng.random() < self.augmentation_config.get('p_octave_shift', 0.0):
            augmentations.append(('octave', lambda s: self._octave_transpose(s, self.augmentation_config['octave_range'])))
        if self.rng.random() < self.augmentation_config.get('p_velocity_scale', 0.0):
            augmentations.append(('velocity', lambda s: self._velocity_scale(s, self.augmentation_config['velocity_scale_range'])))
        if self.rng.random() < self.augmentation_config.get('p_time_warp', 0.0):
            augmentations.append(('time', lambda s: self._time_warp(s, self.augmentation_config['time_warp_range'])))
        
        # Shuffle and apply augmentations
        self.rng.shuffle(augmentations)
        for aug_name, aug_func in augmentations:
            try:
                seq_np = aug_func(seq_np)
            except Exception as e:
                logger.warning(f"Augmentation {aug_name} failed: {e}")
        
        result = torch.from_numpy(seq_np)
        result = torch.clamp(result, min=0, max=self.max_vocab_id)
        
        return result

    def _octave_transpose(self, seq: np.ndarray, octave_range: Tuple[int, int]) -> np.ndarray:
        """Transpose notes by octaves"""
        octave_shift = self.rng.randint(octave_range[0], octave_range[1] + 1)
        if octave_shift == 0:
            return seq
        
        pitch_shift = octave_shift * 12
        pitch_mappings = {}
        
        for i in range(len(seq)):
            token = self.inv_vocab.get(seq[i], '')
            match_on = self.note_on_pattern.match(token)
            
            if match_on:
                hand = match_on.group(1)
                pitch = int(match_on.group(2))
                new_pitch = pitch + pitch_shift
                
                if self.MIN_MIDI_PITCH <= new_pitch <= self.MAX_MIDI_PITCH:
                    new_token = f"{hand}_NOTE_ON_{new_pitch}"
                    if new_token in self.vocab:
                        new_id = self.vocab[new_token]
                        if 0 <= new_id <= self.max_vocab_id:
                            seq[i] = new_id
                            pitch_mappings[(hand, pitch)] = new_pitch
            
            match_off = self.note_off_pattern.match(token)
            if match_off:
                hand = match_off.group(1)
                pitch = int(match_off.group(2))
                
                if (hand, pitch) in pitch_mappings:
                    new_pitch = pitch_mappings[(hand, pitch)]
                    new_token = f"{hand}_NOTE_OFF_{new_pitch}"
                    if new_token in self.vocab:
                        new_id = self.vocab[new_token]
                        if 0 <= new_id <= self.max_vocab_id:
                            seq[i] = new_id
        
        return seq

    def _velocity_scale(self, seq: np.ndarray, scale_range: Tuple[float, float]) -> np.ndarray:
        """Scale velocity values"""
        scale = self.rng.uniform(scale_range[0], scale_range[1])
        
        for i in range(len(seq)):
            token = self.inv_vocab.get(seq[i], '')
            match = self.velocity_pattern.match(token)
            
            if match:
                vel_level = int(match.group(1))
                new_level = int(vel_level * scale)
                new_level = np.clip(new_level, 0, 15)
                new_token = f"VELOCITY_{new_level}"
                
                if new_token in self.vocab:
                    new_id = self.vocab[new_token]
                    if 0 <= new_id <= self.max_vocab_id:
                        seq[i] = new_id
        
        return seq

    def _time_warp(self, seq: np.ndarray, warp_range: Tuple[float, float]) -> np.ndarray:
        """Warp time shift values"""
        warp_factor = self.rng.uniform(warp_range[0], warp_range[1])
        
        for i in range(len(seq)):
            token = self.inv_vocab.get(seq[i], '')
            match = self.time_shift_pattern.match(token)
            
            if match:
                time_value = int(match.group(1))
                new_value = int(time_value * warp_factor)
                new_value = np.clip(new_value, 1, 1000)
                new_token = f"TIME_SHIFT_{new_value}"
                
                if new_token in self.vocab:
                    new_id = self.vocab[new_token]
                    if 0 <= new_id <= self.max_vocab_id:
                        seq[i] = new_id
        
        return seq

def verify_smart_embedding(model, vocab, device='cpu', sample_size=10):
    """
    Verify that Smart Embedding is working correctly
    
    Args:
        model: Model to verify
        vocab: Vocabulary dictionary
        device: Device
        sample_size: Number of tokens to test
    
    Returns:
        bool: Verification success
    """
    # Find RH and LH note tokens
    rh_tokens = []
    lh_tokens = []
    
    for token, token_id in vocab.items():
        if 'RH' in token and 'NOTE_ON' in token:
            rh_tokens.append(token_id)
        elif 'LH' in token and 'NOTE_ON' in token:
            lh_tokens.append(token_id)
        
        if len(rh_tokens) >= sample_size and len(lh_tokens) >= sample_size:
            break
    
    if not rh_tokens or not lh_tokens:
        logger.warning("Not enough RH/LH tokens for verification")
        return True  # Warning only, continue
    
    # Test input generation (same pitch, different hands)
    test_pairs = []
    for token_str, token_id in vocab.items():
        if 'RH_NOTE_ON_60' in token_str:
            rh_60 = token_id
        elif 'LH_NOTE_ON_60' in token_str:
            lh_60 = token_id
    
    if 'rh_60' in locals() and 'lh_60' in locals():
        test_input = torch.tensor([[rh_60, lh_60]]).to(device)
        
        # Get embeddings
        model.eval()
        with torch.no_grad():
            embeddings = model.embedding(test_input)
        
        # Same pitch (60) so shouldn't be completely different,
        # Different hands so shouldn't be completely same
        emb_rh = embeddings[0, 0]
        emb_lh = embeddings[0, 1]
        
        # Calculate cosine similarity
        cos_sim = F.cosine_similarity(emb_rh.unsqueeze(0), emb_lh.unsqueeze(0)).item()
        
        # Similarity should be between 0.3 ~ 0.9 (same pitch, different hands)
        if cos_sim < 0.2:
            logger.warning(f"⚠️ RH/LH embeddings too different (cos_sim={cos_sim:.3f})")
            logger.warning("Pitch information might not be shared properly")
        elif cos_sim > 0.95:
            logger.error(f"❌ RH/LH embeddings too similar (cos_sim={cos_sim:.3f})")
            logger.error("Hand information might not be encoded properly")
            return False
        else:
            logger.info(f"✅ Smart Embedding working correctly (cos_sim={cos_sim:.3f})")
            logger.info("  - Same pitch → partially similar ✓")
            logger.info("  - Different hands → partially different ✓")
    else:
        logger.info("Skipping detailed verification (no NOTE_ON_60 found)")
    
    return True

def focal_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    alpha: float = 0.25,
    reduction: str = 'none'
) -> torch.Tensor:
    ce_loss = F.cross_entropy(logits, targets, reduction='none')
    pt = torch.exp(-ce_loss)
    focal_term = (1 - pt) ** gamma
    focal_loss = alpha * focal_term * ce_loss
    if reduction == 'mean':
        return focal_loss.mean()
    elif reduction == 'sum':
        return focal_loss.sum()
    else:
        return focal_loss

def calculate_weighted_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pad_token_id: int,
    pos_unknown_token_id: int,
    transition_token_ids: Optional[Set[int]] = None,
    transition_weight: float = 1.0,
    reduction: str = 'mean',
    use_focal_loss: bool = False,
    focal_gamma: float = 2.0,
    focal_alpha: float = 0.25,
    label_smoothing: float = 0.0,
    transition_focal_gamma: Optional[float] = None
) -> torch.Tensor:
    batch_size, seq_len, vocab_size = logits.shape
    logits_flat = logits.reshape(-1, vocab_size)
    targets_flat = targets.reshape(-1)
    ignore_mask = (targets_flat == pad_token_id) | (targets_flat == pos_unknown_token_id)
    
    if use_focal_loss:
        if label_smoothing > 0:
            logger.warning("Label smoothing is ignored when using focal loss")
        
        if transition_token_ids and transition_focal_gamma is not None and transition_focal_gamma != focal_gamma:
            trans_mask = torch.zeros_like(targets_flat, dtype=torch.bool)
            for token_id in transition_token_ids:
                trans_mask |= (targets_flat == token_id)
            
            loss_unreduced = torch.zeros_like(targets_flat, dtype=torch.float32)
            
            if trans_mask.any():
                loss_unreduced[trans_mask] = focal_cross_entropy(
                    logits_flat[trans_mask],
                    targets_flat[trans_mask],
                    gamma=transition_focal_gamma,
                    alpha=focal_alpha,
                    reduction='none'
                )
            
            non_trans_mask = ~trans_mask & ~ignore_mask
            if non_trans_mask.any():
                loss_unreduced[non_trans_mask] = focal_cross_entropy(
                    logits_flat[non_trans_mask],
                    targets_flat[non_trans_mask],
                    gamma=focal_gamma,
                    alpha=focal_alpha,
                    reduction='none'
                )
        else:
            loss_unreduced = focal_cross_entropy(
                logits_flat,
                targets_flat,
                gamma=focal_gamma,
                alpha=focal_alpha,
                reduction='none'
            )
    else:
        loss_unreduced = F.cross_entropy(
            logits_flat,
            targets_flat,
            reduction='none',
            label_smoothing=label_smoothing
        )
    
    weights = torch.ones_like(targets_flat, dtype=torch.float32)
    
    if transition_token_ids and transition_weight != 1.0:
        for token_id in transition_token_ids:
            transition_mask = targets_flat == token_id
            weights[transition_mask] = transition_weight
    
    weights[ignore_mask] = 0.0
    weighted_loss = loss_unreduced * weights
    
    if reduction == 'none':
        return weighted_loss.reshape(batch_size, seq_len)
    elif reduction == 'sum':
        return weighted_loss.sum()
    elif reduction == 'mean':
        valid_count = (~ignore_mask).sum()
        if valid_count > 0:
            return weighted_loss.sum() / valid_count
        else:
            return torch.tensor(0.0, device=logits.device)
    else:
        raise ValueError(f"Invalid reduction: {reduction}")

def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: Optional[GradScaler],
    device: torch.device,
    epoch: int,
    accumulation_steps: int = 1,
    pad_token_id: int = 0,
    pos_unknown_token_id: int = 0,
    current_step: int = 0,
    transition_token_ids: Optional[Set[int]] = None,
    transition_weight: float = 2.0,
    use_amp: bool = True,
    use_bf16: bool = False,  # Added BF16 parameter
    use_focal_loss: bool = False,
    focal_gamma: float = 2.0,
    focal_alpha: float = 0.25,
    label_smoothing: float = 0.0,
    transition_focal_gamma: Optional[float] = None,
    inv_vocab: Optional[Dict[int, str]] = None
) -> Tuple[Dict[str, float], int]:
    model.train()
    
    if hasattr(train_loader.dataset, 'set_epoch'):
        train_loader.dataset.set_epoch(epoch)
    
    total_loss = 0.0
    total_tokens = 0
    total_transition_loss = 0.0
    total_transition_tokens = 0
    total_transition_correct = 0
    batch_count = 0
    
    progress_bar = tqdm(train_loader, desc=f"Training epoch {epoch}", leave=True)
    optimizer.zero_grad()
    
    for batch_idx, (x, y, style_ids) in enumerate(progress_bar):
        x = x.to(device)
        y = y.to(device)
        style_ids = style_ids.to(device)
        
        if use_amp:
            # BF16 or FP16 selection
            dtype = torch.bfloat16 if use_bf16 else torch.float16
            
            with torch.cuda.amp.autocast(dtype=dtype):
                logits, _ = model(x, style_ids)
                loss = calculate_weighted_loss(
                    logits, y, pad_token_id, pos_unknown_token_id,
                    transition_token_ids, transition_weight,
                    use_focal_loss=use_focal_loss,
                    focal_gamma=focal_gamma,
                    focal_alpha=focal_alpha,
                    label_smoothing=label_smoothing,
                    transition_focal_gamma=transition_focal_gamma
                ) / accumulation_steps
            
            if scaler is not None:  # FP16 case only
                scaler.scale(loss).backward()
            else:  # BF16 case (no scaler needed)
                loss.backward()
        else:
            # AMP disabled
            logits, _ = model(x, style_ids)
            loss = calculate_weighted_loss(
                logits, y, pad_token_id, pos_unknown_token_id,
                transition_token_ids, transition_weight,
                use_focal_loss=use_focal_loss,
                focal_gamma=focal_gamma,
                focal_alpha=focal_alpha,
                label_smoothing=label_smoothing,
                transition_focal_gamma=transition_focal_gamma
            ) / accumulation_steps
            loss.backward()
        
        with torch.no_grad():
            valid_mask = (y != pad_token_id) & (y != pos_unknown_token_id)
            batch_tokens = valid_mask.sum().item()
            
            if batch_tokens > 0:
                batch_loss_unreduced = calculate_weighted_loss(
                    logits, y, pad_token_id, pos_unknown_token_id,
                    transition_token_ids, transition_weight,
                    reduction='none',
                    use_focal_loss=use_focal_loss,
                    focal_gamma=focal_gamma,
                    focal_alpha=focal_alpha,
                    label_smoothing=label_smoothing,
                    transition_focal_gamma=transition_focal_gamma
                )
                batch_loss = batch_loss_unreduced[valid_mask].mean().item()
                total_loss += batch_loss * batch_tokens
                total_tokens += batch_tokens
                
                if transition_token_ids:
                    trans_mask = torch.zeros_like(y, dtype=torch.bool)
                    for token_id in transition_token_ids:
                        trans_mask |= (y == token_id)
                    trans_mask &= valid_mask
                    trans_tokens = trans_mask.sum().item()
                    
                    if trans_tokens > 0:
                        trans_loss = F.cross_entropy(
                            logits.reshape(-1, logits.size(-1)),
                            y.reshape(-1),
                            reduction='none'
                        ).reshape(y.shape)[trans_mask].mean().item()
                        total_transition_loss += trans_loss * trans_tokens
                        total_transition_tokens += trans_tokens
                        
                        trans_preds = logits[trans_mask].argmax(dim=-1)
                        trans_correct = (trans_preds == y[trans_mask]).sum().item()
                        total_transition_correct += trans_correct
                        
                        if batch_idx == 0:
                            logger.info(f"Epoch {epoch}: Trans mask count: {trans_mask.sum()}")
                            logger.info(f"Trans preds sample: {trans_preds[:10].tolist()}")
                            logger.info(f"Trans targets sample: {y[trans_mask][:10].tolist()}")
                            if trans_mask.sum() > 0:
                                trans_acc_batch = (trans_preds == y[trans_mask]).float().mean().item()
                                logger.info(f"Batch trans accuracy: {trans_acc_batch:.4f}")
                                if trans_acc_batch == 0 and inv_vocab is not None:
                                    logger.warning("All trans predictions wrong! Checking vocab...")
                                    for i in range(min(5, trans_preds.size(0))):
                                        pred_token = inv_vocab.get(trans_preds[i].item(), 'UNKNOWN')
                                        true_token = inv_vocab.get(y[trans_mask][i].item(), 'UNKNOWN')
                                        logger.warning(f"Pred: {pred_token} vs True: {true_token}")
                
                batch_count += 1
        
        # Optimizer step
        if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
            if use_amp and scaler is not None:  # FP16
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:  # BF16 or FP32
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            if TRANSFORMERS_AVAILABLE and hasattr(scheduler, 'step'):
                scheduler.step()
            
            optimizer.zero_grad()
            current_step += 1
        
        if batch_count > 0 and total_tokens > 0:
            avg_loss = total_loss / total_tokens
            progress_dict = {
                'loss': f"{avg_loss:.4f}",
                'ppl': f"{np.exp(avg_loss):.2f}",
                'lr': f"{optimizer.param_groups[0]['lr']:.6f}",
                'step': current_step
            }
            if total_transition_tokens > 0:
                avg_trans_loss = total_transition_loss / total_transition_tokens
                trans_acc = total_transition_correct / total_transition_tokens
                progress_dict['trans_ppl'] = f"{np.exp(avg_trans_loss):.2f}"
                progress_dict['trans_acc'] = f"{trans_acc:.3f}"
            progress_bar.set_postfix(progress_dict)
    
    metrics = {}
    if total_tokens > 0:
        avg_loss = total_loss / total_tokens
        metrics['train_loss'] = avg_loss
        metrics['train_perplexity'] = np.exp(avg_loss)
        metrics['train_tokens'] = total_tokens
    
    if total_transition_tokens > 0:
        avg_transition_loss = total_transition_loss / total_transition_tokens
        metrics['train_transition_loss'] = avg_transition_loss
        metrics['train_transition_perplexity'] = np.exp(avg_transition_loss)
        metrics['train_transition_tokens'] = total_transition_tokens
        metrics['train_transition_accuracy'] = total_transition_correct / total_transition_tokens
        metrics['train_transition_ratio'] = total_transition_tokens / total_tokens
    
    return metrics, current_step

@torch.no_grad()
def validate_epoch(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    use_bf16: bool = False,  # Added BF16 parameter
    pad_token_id: int = 0,
    pos_unknown_token_id: int = 0,
    transition_token_ids: Optional[Set[int]] = None,
    transition_weight: float = 1.0,
    use_focal_loss: bool = False,
    focal_gamma: float = 2.0,
    focal_alpha: float = 0.25,
    label_smoothing: float = 0.0,
    transition_focal_gamma: Optional[float] = None
) -> Dict[str, float]:
    model.eval()
    
    total_loss = 0.0
    total_tokens = 0
    total_transition_loss = 0.0
    total_transition_tokens = 0
    total_transition_correct = 0
    style_losses = defaultdict(float)
    style_counts = defaultdict(int)
    
    progress_bar = tqdm(val_loader, desc="Validating", leave=True)
    
    for x, y, style_ids in progress_bar:
        x = x.to(device)
        y = y.to(device)
        style_ids = style_ids.to(device)
        
        # BF16 support added
        if use_bf16:
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                logits, _ = model(x, style_ids)
                logits = logits.float()  # Convert to FP32 for loss calculation
        else:
            logits, _ = model(x, style_ids)
        
        loss_unreduced = calculate_weighted_loss(
            logits, y, pad_token_id, pos_unknown_token_id,
            transition_token_ids, transition_weight,
            reduction='none',
            use_focal_loss=use_focal_loss,
            focal_gamma=focal_gamma,
            focal_alpha=focal_alpha,
            label_smoothing=label_smoothing,
            transition_focal_gamma=transition_focal_gamma
        )
        
        valid_mask = (y != pad_token_id) & (y != pos_unknown_token_id)
        valid_loss = loss_unreduced[valid_mask]
        
        if valid_loss.numel() > 0:
            total_loss += valid_loss.sum().item()
            total_tokens += valid_loss.numel()
        
        if transition_token_ids:
            trans_mask = torch.zeros_like(y, dtype=torch.bool)
            for token_id in transition_token_ids:
                trans_mask |= (y == token_id)
            trans_mask &= valid_mask
            
            if trans_mask.any():
                trans_loss_unweighted = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    y.reshape(-1),
                    reduction='none'
                ).reshape(y.shape)[trans_mask]
                total_transition_loss += trans_loss_unweighted.sum().item()
                total_transition_tokens += trans_loss_unweighted.numel()
                
                trans_preds = logits[trans_mask].argmax(dim=-1)
                trans_correct = (trans_preds == y[trans_mask]).sum().item()
                total_transition_correct += trans_correct
        
        for i, style_id in enumerate(style_ids):
            style_id = style_id.item()
            style_mask = valid_mask[i]
            if style_mask.any():
                style_loss = loss_unreduced[i][style_mask]
                style_losses[style_id] += style_loss.sum().item()
                style_counts[style_id] += style_loss.numel()
    
    metrics = {}
    
    if total_tokens > 0:
        avg_loss = total_loss / total_tokens
        metrics['val_loss'] = avg_loss
        metrics['val_perplexity'] = np.exp(avg_loss)
        metrics['val_tokens'] = total_tokens
    
    style_perplexities = {}
    for style_id in style_losses:
        if style_counts[style_id] > 0:
            style_avg_loss = style_losses[style_id] / style_counts[style_id]
            style_perplexities[style_id] = np.exp(style_avg_loss)
    metrics['style_perplexities'] = style_perplexities
    
    if total_transition_tokens > 0:
        avg_transition_loss = total_transition_loss / total_transition_tokens
        metrics['val_transition_loss'] = avg_transition_loss
        metrics['val_transition_perplexity'] = np.exp(avg_transition_loss)
        metrics['val_transition_tokens'] = total_transition_tokens
        metrics['val_transition_accuracy'] = total_transition_correct / total_transition_tokens
        if total_tokens > 0:
            metrics['val_transition_ratio'] = total_transition_tokens / total_tokens
    
    return metrics

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epoch: int,
    best_val_loss: float,
    checkpoint_path: Path,
    vocab: Dict[str, int],
    inv_vocab: Dict[int, str],
    config: Dict[str, Any],
    global_step: int = 0,
    is_best: bool = False,
    wandb_run_id: Optional[str] = None,
    train_loss: Optional[float] = None,
    val_loss: Optional[float] = None
):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'epoch': epoch,
        'global_step': global_step,
        'best_val_loss': best_val_loss,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'config': config,
        'vocab': vocab,
        'inv_vocab': inv_vocab,
        'vocab_size': len(vocab),
        'max_vocab_id': max(vocab.values()),
        'model_size': config.get('model_size', 'base'),
        'pytorch_version': torch.__version__,
        'timestamp': datetime.now().isoformat(),
        'wandb_run_id': wandb_run_id,
        'rng_state': {
            'python': random.getstate(),
            'numpy': np.random.get_state(),
            'torch': torch.get_rng_state().cpu().numpy().tolist(),
            'cuda': [state.cpu().numpy().tolist() for state in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else None
        }
    }
    
    checkpoint_file = checkpoint_path / f'checkpoint_epoch_{epoch}.pt'
    torch.save(checkpoint, checkpoint_file)
    logger.info(f"Saved complete checkpoint to {checkpoint_file}")
    
    if is_best:
        best_file = checkpoint_path / 'best_model.pt'
        torch.save(checkpoint, best_file)
        logger.info(f"Saved best model to {best_file}")
        
        weights_file = checkpoint_path / 'best_model_weights.pt'
        torch.save(model.state_dict(), weights_file)
        
        if wandb.run is not None:
            artifact = wandb.Artifact(
                name=f"model-{wandb.run.id}",
                type="model",
                description=f"Best model with val_loss={best_val_loss:.4f}"
            )
            artifact.add_file(str(best_file))
            wandb.log_artifact(artifact)
    
    # Keep only last N checkpoints
    keep_last_n = 3
    checkpoints = sorted(checkpoint_path.glob('checkpoint_epoch_*.pt'))
    if len(checkpoints) > keep_last_n:
        for old_checkpoint in checkpoints[:-keep_last_n]:
            old_checkpoint.unlink()
            logger.info(f"Removed old checkpoint: {old_checkpoint}")

def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer = None,
    scheduler: torch.optim.lr_scheduler._LRScheduler = None,
    device: torch.device = None
) -> Tuple[int, int, float, Optional[str]]:
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if TRANSFORMERS_AVAILABLE and hasattr(scheduler, '_step_count'):
            scheduler._step_count = checkpoint.get('global_step', 0)
    
    # Restore RNG states
    if 'rng_state' in checkpoint:
        try:
            rng_states = checkpoint['rng_state']
            if rng_states.get('python') is not None:
                random.setstate(rng_states['python'])
            if rng_states.get('numpy') is not None:
                np.random.set_state(rng_states['numpy'])
            if rng_states.get('torch') is not None:
                torch_state = rng_states['torch']
                if isinstance(torch_state, list):
                    torch_state = torch.ByteTensor(torch_state)
                elif hasattr(torch_state, 'cpu'):
                    torch_state = torch_state.cpu()
                torch.set_rng_state(torch_state)
            if rng_states.get('cuda') is not None and torch.cuda.is_available():
                cuda_states = rng_states['cuda']
                if isinstance(cuda_states[0], list):
                    cuda_states = [torch.ByteTensor(state) for state in cuda_states]
                torch.cuda.set_rng_state_all(cuda_states)
        except Exception as e:
            logger.warning(f"Could not restore RNG states: {e}. This is usually fine.")
    
    epoch = checkpoint.get('epoch', 0)
    global_step = checkpoint.get('global_step', 0)
    best_val_loss = checkpoint.get('best_val_loss', float('inf'))
    wandb_run_id = checkpoint.get('wandb_run_id', None)
    
    logger.info(f"Resumed from epoch {epoch}, global step {global_step}, best val loss {best_val_loss:.4f}")
    
    return epoch, global_step, best_val_loss, wandb_run_id

def main():
    parser = argparse.ArgumentParser(description='Train Conditional Music Transformer with RoPE')
    
    # Data arguments
    parser.add_argument('--data_path', type=str, default='output/transformer_dataset.pt',
                        help='Path to the dataset file')
    
    # Model arguments
    parser.add_argument('--model_size', type=str, default='base',
                        choices=['small', 'base', 'large'],
                        help='Model size configuration')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size per GPU')
    parser.add_argument('--accumulation_steps', type=int, default=4,
                        help='Gradient accumulation steps')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--warmup_steps', type=int, default=1000,
                        help='Number of warmup steps')
    parser.add_argument('--patience', type=int, default=15,
                        help='Early stopping patience')
    
    # BF16 support argument
    parser.add_argument('--use_bf16', action='store_true',
                        help='Use bfloat16 precision (for FlashAttention compatibility)')
    
    # Fine-tuning arguments
    parser.add_argument('--finetune_from', type=str, default=None,
                        help='Path to pretrained model checkpoint for fine-tuning')
    parser.add_argument('--freeze_ratio', type=float, default=0.8,
                        help='Ratio of transformer layers to freeze from bottom up')
    
    # Transition token arguments
    parser.add_argument('--transition_weight', type=float, default=2.0,
                        help='Weight multiplier for transition tokens in loss')
    parser.add_argument('--disable_transition_weighting', action='store_true',
                        help='Disable transition token weighting')
    
    # Loss function arguments
    parser.add_argument('--use_focal_loss', action='store_true',
                        help='Use focal loss for handling class imbalance')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                        help='Gamma parameter for focal loss')
    parser.add_argument('--focal_alpha', type=float, default=0.25,
                        help='Alpha parameter for focal loss')
    parser.add_argument('--transition_focal_gamma', type=float, default=None,
                        help='Specific gamma for transition tokens')
    parser.add_argument('--label_smoothing', type=float, default=0.0,
                        help='Label smoothing factor')
    
    # Augmentation arguments (safe defaults)
    parser.add_argument('--p_augment', type=float, default=0.0,
                        help='Probability of applying augmentation')
    parser.add_argument('--p_octave_shift', type=float, default=0.0,
                        help='Probability of octave transposition')
    parser.add_argument('--p_velocity_scale', type=float, default=0.0,
                        help='Probability of velocity scaling')
    parser.add_argument('--p_time_warp', type=float, default=0.0,
                        help='Probability of time warping')
    
    # Checkpoint arguments
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    # Wandb arguments
    parser.add_argument('--wandb_project', type=str, default='music-transformer',
                        help='Wandb project name')
    parser.add_argument('--wandb_run_name', type=str, default=None,
                        help='Wandb run name')
    parser.add_argument('--wandb_run_id', type=str, default=None,
                        help='Wandb run ID for resuming')
    
    parser.add_argument('--max_seq_length', type=int, default=2048,
                        help='Maximum sequence length (truncate longer sequences)')
    
    # Other arguments
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--no_wandb', action='store_true',
                        help='Disable wandb logging')
    parser.add_argument('--low_memory', action='store_true',
                        help='Enable low memory mode')
    parser.add_argument('--force_random_split', action='store_true',
                        help='Force random split')
    parser.add_argument('--disable_amp', action='store_true',
                        help='Disable automatic mixed precision')
    
    args = parser.parse_args()
    
    # Set transition_focal_gamma if not specified
    if args.transition_focal_gamma is None:
        args.transition_focal_gamma = args.focal_gamma
    
    # Validate arguments
    if args.batch_size <= 0:
        raise ValueError("Batch size must be positive")
    if args.accumulation_steps <= 0:
        raise ValueError("Accumulation steps must be positive")
    if not 0 <= args.p_augment <= 1:
        raise ValueError("p_augment must be between 0 and 1")
    if not 0 <= args.freeze_ratio <= 1:
        raise ValueError("freeze_ratio must be between 0 and 1")
    if args.focal_gamma <= 0:
        raise ValueError("focal_gamma must be positive")
    if args.focal_alpha <= 0 or args.focal_alpha > 1:
        raise ValueError("focal_alpha must be between 0 and 1")
    if not 0 <= args.label_smoothing < 1:
        raise ValueError("label_smoothing must be between 0 and 1")
    if args.transition_focal_gamma is not None and args.transition_focal_gamma <= 0:
        raise ValueError("transition_focal_gamma must be positive")
    
    # Check conflicting arguments
    if args.finetune_from and args.resume_from_checkpoint:
        raise ValueError("Cannot use both --finetune_from and --resume_from_checkpoint")
    
    # Adjust settings for fine-tuning
    if args.finetune_from:
        if args.lr >= 3e-4:
            args.lr = 1e-5
            logger.info(f"Using fine-tuning learning rate: {args.lr}")
        if args.epochs == 100:
            args.epochs = 50
            logger.info(f"Using fine-tuning epochs: {args.epochs}")
        if args.warmup_steps == 1000:
            args.warmup_steps = 200
            logger.info(f"Using fine-tuning warmup steps: {args.warmup_steps}")
    
    # Adjust for low memory mode
    if args.low_memory:
        args.batch_size = max(1, args.batch_size // 4)
        logger.info(f"Low memory mode: reduced batch size to {args.batch_size}")
    
    # Set random seed
    set_seed(args.seed)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Setup checkpoint directory
    checkpoint_path = Path(args.checkpoint_dir)
    if args.finetune_from:
        checkpoint_path = checkpoint_path.parent / f"{checkpoint_path.name}_finetune"
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize wandb
    wandb_run_id = args.wandb_run_id
    if not args.no_wandb:
        run_name = args.wandb_run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        if args.finetune_from:
            run_name = f"finetune_{run_name}"
        
        config = vars(args).copy()
        config['is_finetune'] = args.finetune_from is not None
        
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config=config,
            resume='allow' if args.resume_from_checkpoint and not args.finetune_from else None,
            id=wandb_run_id
        )
        wandb_run_id = wandb.run.id
    
    try:
        # Load dataset
        logger.info(f"Loading dataset from {args.data_path}")
        if not Path(args.data_path).exists():
            raise FileNotFoundError(f"Dataset file not found: {args.data_path}")
        
        data = torch.load(args.data_path, map_location='cpu', weights_only=False)
        
        # Validate dataset
        required_keys = ['sequences', 'vocab', 'inv_vocab', 'metadata', 'config']
        for key in required_keys:
            if key not in data:
                raise KeyError(f"Dataset missing required key: {key}")
        
        sequences = data['sequences']
        vocab = data['vocab']
        inv_vocab = data['inv_vocab']
        metadata = data['metadata']
        config = data['config']
        config['model_size'] = args.model_size
        
        logger.info(f"Loaded {len(sequences)} sequences, vocab size: {len(vocab)}")
        
        # Handle non-consecutive vocabulary
        max_vocab_id = max(vocab.values())
        logger.info(f"Max vocab ID: {max_vocab_id}")
        if max_vocab_id + 1 > len(vocab):
            logger.warning(f"Non-consecutive vocab detected! Using vocab_size = {max_vocab_id + 1} for model")
        vocab_size_for_model = max_vocab_id + 1
        
        # Get special token IDs
        pad_token_id = vocab['<PAD>']
        pos_unknown_token_id = vocab.get('<POS_UNKNOWN>', vocab.get('<UNK>', pad_token_id))
        
        # Get transition token IDs
        transition_token_ids = None
        if not args.disable_transition_weighting:
            transition_token_ids = set(config.get('transition_token_ids', []))
            if transition_token_ids:
                logger.info(f"Found {len(transition_token_ids)} transition tokens")
                logger.info(f"Transition weight: {args.transition_weight}x")
                trans_tokens = [inv_vocab.get(tid, f"<ID:{tid}>") for tid in list(transition_token_ids)[:5]]
                logger.info(f"Example transition tokens: {trans_tokens}")
            else:
                logger.info("No transition tokens found in dataset")
                if args.finetune_from:
                    logger.warning("Fine-tuning but no transition tokens found")
        
        # Extract style IDs for stratification
        temp_dataset = MusicDataset(sequences, vocab, inv_vocab, metadata, seed=args.seed)
        all_style_ids = temp_dataset.style_ids
        style_counts = Counter(all_style_ids)
        logger.info(f"Style distribution: {dict(sorted(style_counts.items()))}")
        logger.info(f"Total styles: {len(style_counts)}, Total sequences: {len(sequences)}")
        
        # Check dataset size
        if len(sequences) < 20:
            raise ValueError(f"Dataset too small for train/val split: only {len(sequences)} sequences")
        
        # Find rare styles
        rare_styles = [style for style, count in style_counts.items() if count < 2]
        if rare_styles:
            logger.warning(f"Found {len(rare_styles)} styles with less than 2 samples: {rare_styles}")
        
        # Split dataset
        indices = list(range(len(sequences)))
        if rare_styles or args.force_random_split:
            if rare_styles:
                logger.info("Using random split due to rare styles")
            else:
                logger.info("Using random split (forced by --force_random_split)")
            
            train_idx, val_idx = train_test_split(
                indices,
                test_size=0.1,
                random_state=args.seed,
                shuffle=True
            )
            
            val_styles = [all_style_ids[i] for i in val_idx]
            val_style_counts = Counter(val_styles)
            logger.info(f"Validation set has {len(val_style_counts)} unique styles out of {len(style_counts)} total")
            
            if len(val_style_counts) < len(style_counts) * 0.5:
                logger.warning(f"Validation set covers only {len(val_style_counts)/len(style_counts)*100:.1f}% of styles")
        else:
            logger.info("Using stratified split")
            train_idx, val_idx = train_test_split(
                indices,
                test_size=0.1,
                stratify=all_style_ids,
                random_state=args.seed
            )
        
        logger.info(f"Train samples: {len(train_idx)}, Val samples: {len(val_idx)}")
        
        # Create augmentation config
        augmentation_config = {
            'p_augment': args.p_augment,
            'p_octave_shift': args.p_octave_shift,
            'p_velocity_scale': args.p_velocity_scale,
            'p_time_warp': args.p_time_warp,
            'octave_range': (-1, 1),
            'velocity_scale_range': (0.8, 1.2),
            'time_warp_range': (0.9, 1.1)
        }
        
        # Log augmentation settings
        if args.p_augment > 0:
            logger.info(f"Augmentation enabled with p={args.p_augment}")
            logger.info(f"  Octave shift: p={args.p_octave_shift}")
            logger.info(f"  Velocity scale: p={args.p_velocity_scale}")
            logger.info(f"  Time warp: p={args.p_time_warp}")
        else:
            logger.info("Augmentation disabled")
        
        # Create datasets
        train_dataset = MusicDataset(
            sequences, vocab, inv_vocab, metadata,
            indices=train_idx,
            augmentation_config=augmentation_config,
            transition_token_ids=transition_token_ids,
            seed=args.seed,
            max_seq_length=args.max_seq_length
        )

        val_dataset = MusicDataset(
            sequences, vocab, inv_vocab, metadata,
            indices=val_idx,
            augmentation_config=None,
            transition_token_ids=transition_token_ids,
            seed=args.seed,
            max_seq_length=args.max_seq_length
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=args.num_workers > 0
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size * 2,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=args.num_workers > 0
        )
        
        # Create model
        logger.info(f"Creating {args.model_size} model with vocab_size {vocab_size_for_model}")

        # Style count safe handling
        n_styles_to_use = max(config.get('max_cluster_id', 0) + 1, 10)
        logger.info(f"Using {n_styles_to_use} style embeddings")

        model = create_model(
            vocab_size=vocab_size_for_model,
            n_styles=n_styles_to_use,
            model_size=args.model_size,
            use_smart_embedding=True
        ).to(device)
        
        model.setup_vocab_mappings(vocab)
        
        # Verify Smart Embedding
        logger.info("Verifying Smart Embedding...")
        if not verify_smart_embedding(model, vocab, device=device):
            logger.warning("Smart Embedding verification failed but continuing anyway")
        
        # Handle fine-tuning
        if args.finetune_from:
            logger.info(f"FINE-TUNING MODE")
            logger.info(f"Loading pretrained model from: {args.finetune_from}")
            logger.info(f"Freeze ratio: {args.freeze_ratio}")
            logger.info(f"Learning rate: {args.lr}")
            
            # Check pretrained model path
            if not Path(args.finetune_from).exists():
                alt_path = Path(args.finetune_from).parent / 'best_model_weights.pt'
                if alt_path.exists():
                    logger.info(f"Using alternative path: {alt_path}")
                    args.finetune_from = str(alt_path)
                else:
                    raise FileNotFoundError(f"Pretrained model not found: {args.finetune_from}")
            
            # Load pretrained weights
            old_checkpoint = torch.load(args.finetune_from, map_location=device, weights_only=False)
            
            if isinstance(old_checkpoint, dict) and 'model_state_dict' in old_checkpoint:
                old_state_dict = old_checkpoint['model_state_dict']
                logger.info("Loaded full checkpoint with model_state_dict")
            elif isinstance(old_checkpoint, dict) and 'token_embedding.weight' in old_checkpoint:
                old_state_dict = old_checkpoint
                logger.info("Loaded state dict directly")
            else:
                old_state_dict = old_checkpoint
                logger.info("Loaded weights directly")
            
            # Handle vocabulary size changes
            old_vocab_size = old_state_dict['token_embedding.weight'].shape[0]
            new_vocab_size = vocab_size_for_model
            logger.info(f"Vocabulary size changed: {old_vocab_size} -> {new_vocab_size}")
            
            if new_vocab_size != old_vocab_size:
                # Resize embeddings
                old_embeddings = old_state_dict['token_embedding.weight']
                new_embeddings = torch.nn.Parameter(torch.empty(new_vocab_size, old_embeddings.shape[1]))
                
                with torch.no_grad():
                    if new_vocab_size > old_vocab_size:
                        # Copy old embeddings
                        new_embeddings[:old_vocab_size] = old_embeddings
                        
                        # Initialize new tokens with distribution similar to old tokens
                        old_mean = old_embeddings.mean(dim=0)
                        old_std = old_embeddings.std(dim=0)
                        
                        for i in range(old_vocab_size, new_vocab_size):
                            new_embeddings[i] = torch.normal(mean=old_mean, std=old_std)
                        
                        logger.info(f"Initialized {new_vocab_size - old_vocab_size} new token embeddings using old distribution")
                        
                        new_tokens = [token for token in vocab.keys() if vocab[token] >= old_vocab_size]
                        logger.info(f"New tokens include: {new_tokens[:10]}{'...' if len(new_tokens) > 10 else ''}")
                    else:
                        new_embeddings = old_embeddings[:new_vocab_size]
                
                old_state_dict['token_embedding.weight'] = new_embeddings
                
                # Resize LM head if present
                if 'lm_head.weight' in old_state_dict:
                    old_lm_head = old_state_dict['lm_head.weight']
                    if old_lm_head.shape[0] == old_vocab_size:
                        new_lm_head = torch.nn.Parameter(torch.empty(new_vocab_size, old_lm_head.shape[1]))
                        
                        with torch.no_grad():
                            if new_vocab_size > old_vocab_size:
                                new_lm_head[:old_vocab_size] = old_lm_head
                                
                                old_lm_mean = old_lm_head.mean(dim=0)
                                old_lm_std = old_lm_head.std(dim=0)
                                
                                for i in range(old_vocab_size, new_vocab_size):
                                    new_lm_head[i] = torch.normal(mean=old_lm_mean, std=old_lm_std)
                                
                                logger.info(f"Initialized new lm_head weights using old distribution")
                            else:
                                new_lm_head = old_lm_head[:new_vocab_size]
                        
                        old_state_dict['lm_head.weight'] = new_lm_head
            
            # Load state dict
            missing_keys, unexpected_keys = model.load_state_dict(old_state_dict, strict=False)
            if missing_keys:
                logger.warning(f"Missing keys in checkpoint: {missing_keys}")
            if unexpected_keys:
                logger.warning(f"Unexpected keys in checkpoint: {unexpected_keys}")
            
            # Always unfreeze embeddings and lm_head for new token learning
            model.token_embedding.requires_grad_(True)
            if hasattr(model, 'lm_head'):
                model.lm_head.requires_grad_(True)
            logger.info("Unfroze embeddings and lm_head for new token learning")
            
            # Freeze layers (but keep embeddings and lm_head trainable)
            if args.freeze_ratio > 0:
                transformer_params = []
                embedding_params = []
                lm_head_params = []
                other_params = []
                
                for name, param in model.named_parameters():
                    if 'blocks' in name and any(f'.{i}.' in name for i in range(model.blocks.__len__())):
                        transformer_params.append((name, param))
                    elif 'embedding' in name:
                        embedding_params.append((name, param))
                    elif 'lm_head' in name:
                        lm_head_params.append((name, param))
                    else:
                        other_params.append((name, param))
                
                # Sort transformer params by layer number
                transformer_params.sort(key=lambda x: int(x[0].split('.')[2]) if x[0].split('.')[2].isdigit() else 0)
                
                num_to_freeze = int(len(transformer_params) * args.freeze_ratio)
                frozen_params = 0
                total_params = 0
                
                # Freeze transformer layers according to freeze_ratio
                for i, (name, param) in enumerate(transformer_params):
                    total_params += param.numel()
                    if i < num_to_freeze:
                        param.requires_grad = False
                        frozen_params += param.numel()
                        logger.debug(f"Froze {name}")
                
                # Count params but keep embeddings and lm_head trainable
                for name, param in embedding_params + lm_head_params:
                    total_params += param.numel()
                    param.requires_grad = True  # Explicitly keep trainable
                    logger.debug(f"Kept {name} trainable")
                
                # Handle other params
                for name, param in other_params:
                    total_params += param.numel()
                
                logger.info(f"Froze {num_to_freeze}/{len(transformer_params)} transformer layers")
                logger.info(f"Embeddings and LM head kept trainable for new token adaptation")
                logger.info(f"Froze {frozen_params:,}/{total_params:,} parameters ({frozen_params/total_params*100:.1f}%)")
                logger.info(f"Trainable parameters: {total_params-frozen_params:,}")
            
            # Adjust learning rate for fine-tuning
            if args.lr > 1e-4:
                old_lr = args.lr
                args.lr = min(args.lr, 1e-4)
                logger.info(f"Reduced learning rate for fine-tuning: {old_lr} -> {args.lr}")
        
        # Log model info
        logger.info(f"Model parameters: {model.get_num_params() / 1e6:.2f}M")
        
        if args.finetune_from:
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in model.parameters())
            logger.info(f"Fine-tuning mode: {trainable_params/1e6:.2f}M/{total_params/1e6:.2f}M trainable parameters ({trainable_params/total_params*100:.1f}%)")
        else:
            logger.info("Training from scratch")
        
        # Log loss settings
        if args.use_focal_loss:
            logger.info(f"Using Focal Loss with gamma={args.focal_gamma}, alpha={args.focal_alpha}")
            if args.transition_focal_gamma != args.focal_gamma:
                logger.info(f"Using special transition gamma={args.transition_focal_gamma} for transition tokens")
        
        if args.label_smoothing > 0:
            logger.info(f"Using Label Smoothing with factor={args.label_smoothing}")
        
        # Create optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.95)
        )
        
        # Create scheduler
        if TRANSFORMERS_AVAILABLE:
            steps_per_epoch = math.ceil(len(train_loader) / args.accumulation_steps)
            total_steps = steps_per_epoch * args.epochs
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=args.warmup_steps,
                num_training_steps=total_steps
            )
            logger.info(f"Using cosine scheduler with {args.warmup_steps} warmup steps")
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=10,
                T_mult=2,
                eta_min=args.lr * 0.1
            )
        
        # Setup mixed precision training with BF16 support
        use_amp = not args.disable_amp and device.type == 'cuda'
        if use_amp:
            if args.use_bf16:
                # BF16 doesn't need GradScaler
                scaler = None
                logger.info("Using bfloat16 precision (no GradScaler needed)")
            else:
                from torch.cuda.amp import GradScaler
                scaler = GradScaler()
                logger.info("Using automatic mixed precision (AMP) with fp16")
        else:
            scaler = None
            logger.info("Mixed precision disabled, using FP32")
        
        # Initialize training state
        start_epoch = 0
        global_step = 0
        best_val_loss = float('inf')
        patience_counter = 0
        
        # Resume from checkpoint
        if args.resume_from_checkpoint:
            if not Path(args.resume_from_checkpoint).exists():
                raise FileNotFoundError(f"Checkpoint not found: {args.resume_from_checkpoint}")
            
            load_optimizer = not args.finetune_from
            start_epoch, global_step, best_val_loss, loaded_wandb_id = load_checkpoint(
                args.resume_from_checkpoint, model,
                optimizer if load_optimizer else None,
                scheduler if load_optimizer else None,
                device
            )
            
            if args.finetune_from:
                logger.info("Fine-tuning mode: resetting epoch and step counters")
                start_epoch = 0
                global_step = 0
                best_val_loss = float('inf')
            
            if loaded_wandb_id and not args.no_wandb and not args.finetune_from:
                wandb_run_id = loaded_wandb_id
            
            patience_counter = 0
        
        # Setup wandb watching
        if not args.no_wandb:
            wandb.watch(model, log='gradients', log_freq=100)
        
        # Training loop
        epoch = start_epoch
        for epoch in range(start_epoch, args.epochs):
            logger.info(f"\nEpoch {epoch + 1}/{args.epochs}")
            
            # Train for one epoch
            train_metrics, global_step = train_epoch(
                model, train_loader, optimizer, scheduler, scaler,
                device, epoch, args.accumulation_steps, pad_token_id,
                pos_unknown_token_id, global_step,
                transition_token_ids, args.transition_weight,
                use_amp=use_amp,
                use_bf16=args.use_bf16,  # Pass BF16 flag
                use_focal_loss=args.use_focal_loss,
                focal_gamma=args.focal_gamma,
                focal_alpha=args.focal_alpha,
                label_smoothing=args.label_smoothing,
                transition_focal_gamma=args.transition_focal_gamma,
                inv_vocab=inv_vocab
            )
            
            # Validate
            val_metrics = validate_epoch(
                model, val_loader, device,
                use_bf16=args.use_bf16,  # Pass BF16 flag
                pad_token_id=pad_token_id,
                pos_unknown_token_id=pos_unknown_token_id,
                transition_token_ids=transition_token_ids,
                transition_weight=args.transition_weight,
                use_focal_loss=args.use_focal_loss,
                focal_gamma=args.focal_gamma,
                focal_alpha=args.focal_alpha,
                label_smoothing=args.label_smoothing,
                transition_focal_gamma=args.transition_focal_gamma
            )
            
            # Step scheduler (if not using transformers)
            if not TRANSFORMERS_AVAILABLE:
                scheduler.step()
            
            # Prepare metrics for logging
            metrics = {
                'epoch': epoch + 1,
                'lr': optimizer.param_groups[0]['lr'],
                'global_step': global_step,
                **train_metrics,
                **val_metrics
            }
            
            # Add style perplexities
            for style_id, ppl in val_metrics.get('style_perplexities', {}).items():
                metrics[f'val_ppl_style_{style_id}'] = ppl
            
            # Log metrics
            log_parts = [
                f"Loss: {train_metrics.get('train_loss', 0):.4f}",
                f"PPL: {train_metrics.get('train_perplexity', 1):.2f}",
                f"Val Loss: {val_metrics.get('val_loss', 0):.4f}",
                f"Val PPL: {val_metrics.get('val_perplexity', 1):.2f}"
            ]
            
            if 'train_transition_perplexity' in train_metrics:
                log_parts.append(f"Trans PPL: {train_metrics['train_transition_perplexity']:.2f}")
            if 'train_transition_accuracy' in train_metrics:
                log_parts.append(f"Trans Acc: {train_metrics['train_transition_accuracy']:.3f}")
            
            logger.info(" | ".join(log_parts))
            
            # Log to wandb
            if not args.no_wandb:
                wandb.log(metrics)
            
            # Check for best model
            current_val_loss = val_metrics.get('val_loss', float('inf'))
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                patience_counter = 0
                
                save_checkpoint(
                    model, optimizer, scheduler, epoch + 1, best_val_loss,
                    checkpoint_path, vocab, inv_vocab, config,
                    global_step, is_best=True, wandb_run_id=wandb_run_id,
                    train_loss=train_metrics.get('train_loss'),
                    val_loss=current_val_loss
                )
                logger.info(f"New best model saved with val_loss: {best_val_loss:.4f}")
            else:
                patience_counter += 1
                logger.info(f"Patience counter: {patience_counter}/{args.patience}")
                
                if patience_counter >= args.patience:
                    logger.info("Early stopping triggered!")
                    break
            
            # Save periodic checkpoint
            if (epoch + 1) % 5 == 0:
                save_checkpoint(
                    model, optimizer, scheduler, epoch + 1, best_val_loss,
                    checkpoint_path, vocab, inv_vocab, config,
                    global_step, is_best=False, wandb_run_id=wandb_run_id,
                    train_loss=train_metrics.get('train_loss'),
                    val_loss=current_val_loss
                )
        
        # Training completed
        logger.info("Training completed!")
        logger.info(f"Best validation loss: {best_val_loss:.4f}")
        
        if args.finetune_from:
            logger.info(f"Fine-tuning completed successfully!")
            logger.info(f"Model adapted from {args.finetune_from} to handle {len(transition_token_ids) if transition_token_ids else 0} transition tokens")
        
        # Save final checkpoint
        save_checkpoint(
            model, optimizer, scheduler, epoch + 1, best_val_loss,
            checkpoint_path, vocab, inv_vocab, config,
            global_step, is_best=False, wandb_run_id=wandb_run_id,
            train_loss=train_metrics.get('train_loss', None) if 'train_metrics' in locals() else None,
            val_loss=val_metrics.get('val_loss', None) if 'val_metrics' in locals() else None
        )
        
    except KeyboardInterrupt:
        logger.info("Training interrupted! Saving checkpoint...")
        if 'model' in locals() and 'optimizer' in locals():
            interrupted_path = checkpoint_path / f'interrupted_epoch{epoch}_step{global_step}.pt'
            checkpoint = {
                'epoch': epoch,
                'global_step': global_step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if 'scheduler' in locals() else None,
                'best_val_loss': best_val_loss if 'best_val_loss' in locals() else float('inf'),
                'wandb_run_id': wandb_run_id,
                'config': config if 'config' in locals() else {},
                'vocab': vocab if 'vocab' in locals() else {},
                'inv_vocab': inv_vocab if 'inv_vocab' in locals() else {},
                'vocab_size': len(vocab) if 'vocab' in locals() else 0,
                'max_vocab_id': max(vocab.values()) if 'vocab' in locals() else 0,
                'model_size': args.model_size
            }
            torch.save(checkpoint, interrupted_path)
            logger.info(f"Saved interrupted checkpoint to {interrupted_path}")
    
    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        raise
    
    finally:
        if not args.no_wandb and wandb.run is not None:
            wandb.finish()

if __name__ == '__main__':
    main()