#!/usr/bin/env python3
"""
Beethoven Generator - Enhanced Flexible Generation Version
- Free generation mode for pure transformer creativity
- Simplified constraints for better musicality
- Template-guided or completely free generation
- Added sampling options and entropy tracking
"""

import torch
import torch.nn.functional as F
import json
import re
import random
import time
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from midiutil import MIDIFile

class BeethovenGeneratorRefined:
    def __init__(self, model, vocab: Dict[str, int], inv_vocab: Dict[int, str],
                 device: torch.device = None, random_seed: Optional[int] = None,
                 free_generation: bool = False, sampling_method: str = 'multinomial',
                 top_k: int = 50, top_p: float = 0.95, temperature: float = 1.0,
                 bias_strength: float = 1.0):
        self.model = model
        self.vocab = vocab
        self.inv_vocab = inv_vocab
        self.device = device or torch.device('cpu')
        self.free_generation = free_generation
        self.sampling_method = sampling_method
        self.top_k = top_k
        self.top_p = top_p
        self.base_temperature = temperature
        self.bias_strength = bias_strength
        
        # Set random seed
        if random_seed is not None:
            self.random_seed = random_seed
            random.seed(random_seed)
            torch.manual_seed(random_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(random_seed)
            print(f"🎲 Random seed: {random_seed} (reproducible)")
        else:
            self.random_seed = random.randint(0, 1000000)
            random.seed(self.random_seed)
            torch.manual_seed(self.random_seed)
            print(f"🎲 Random seed: {self.random_seed} (unique each run)")
        
        self.TICKS_PER_BEAT = 480
        self.TICKS_PER_BAR = self.TICKS_PER_BEAT * 4
        
        # Simplified pitch ranges
        self.RH_MIN = 55  # G3
        self.RH_MAX = 84  # C6
        self.LH_MIN = 36  # C2
        self.LH_MAX = 60  # C4
        
        # Track state
        self.last_rh_pitch = 67
        self.last_lh_pitch = 48
        self.current_harmony = None
        self.current_key_root = 60
        self.is_minor_key = False
        
        # Simplified tracking
        self.active_notes = {}
        self.last_action_tick = 0
        self.recent_note_count = 0
        
        # Entropy tracking
        self.entropy_values = []
        
        # Harmony mappings (keep for template mode)
        self.ROMAN_TO_INTERVAL = {
            'I': 0, 'II': 2, 'III': 4, 'IV': 5, 'V': 7, 'VI': 9, 'VII': 11,
            'i': 0, 'ii': 2, 'iii': 3, 'iv': 5, 'v': 7, 'vi': 8, 'vii': 10,
            'ii°': 2, 'vii°': 10, 'V7': 7, 'N': 1,
        }
        
        self.CHORD_STRUCTURES = {
            'major': [0, 4, 7],
            'minor': [0, 3, 7],
            'dim': [0, 3, 6],
            'dom7': [0, 4, 7, 10],
            'aug': [0, 4, 8],
            'maj7': [0, 4, 7, 11],
            'min7': [0, 3, 7, 10],
            'half-dim': [0, 3, 6, 10],
        }
    
    def _sample_token(self, logits: torch.Tensor, temperature: float = 1.0) -> Tuple[int, torch.Tensor]:
        """Advanced sampling with multiple methods"""
        
        # Apply temperature
        logits = logits / temperature
        probs = F.softmax(logits, dim=-1)
        
        # Calculate entropy for tracking
        entropy = -torch.sum(probs * torch.log(probs + 1e-10))
        self.entropy_values.append(entropy.item())
        
        if self.sampling_method == 'argmax':
            # Deterministic - highest probability
            next_token = torch.argmax(probs).item()
        elif self.sampling_method == 'top_k':
            # Top-k sampling
            top_k_probs, top_k_indices = torch.topk(probs, min(self.top_k, probs.size(-1)))
            top_k_probs = top_k_probs / top_k_probs.sum()
            next_token = top_k_indices[torch.multinomial(top_k_probs, 1)].item()
        elif self.sampling_method == 'nucleus' or self.sampling_method == 'top_p':
            # Nucleus (top-p) sampling
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            
            # Find cutoff
            cutoff_idx = torch.searchsorted(cumulative_probs, self.top_p).item() + 1
            cutoff_idx = min(cutoff_idx, probs.size(-1))
            
            # Sample from nucleus
            nucleus_probs = sorted_probs[:cutoff_idx]
            nucleus_probs = nucleus_probs / nucleus_probs.sum()
            idx = torch.multinomial(nucleus_probs, 1).item()
            next_token = sorted_indices[idx].item()
        else:  # multinomial (default)
            next_token = torch.multinomial(probs, 1).item()
        
        return next_token, probs
    
    def generate(self, template: Dict, max_length: int = 768) -> List[int]:
        """Generate music with flexible constraints"""
        
        seed = template.get('seed_tokens', [])
        waypoints = template.get('waypoints', {})
        rhythm_waypoints = template.get('rhythm_waypoints', {})
        key_sig = template.get('metadata', {}).get('key_signature', 'C')
        
        # Parse key
        self.current_key_root = self._parse_key_root(key_sig)
        self.is_minor_key = 'm' in key_sig.lower()
        
        # Check for free generation
        if not seed and not waypoints:
            print("\n🎵 Full free generation - no seed or template")
            self.free_generation = True
        elif self.free_generation:
            print("\n🎵 Free generation mode - minimal constraints after seed")
        else:
            print("\n🎵 Template-guided generation")
        
        generated = seed.copy() if seed else []
        
        # Calculate target
        seed_bars = self._estimate_bars(seed) if seed else 0
        template_bars = len(waypoints) if waypoints else 8
        target_bars = seed_bars + template_bars
        
        print(f"📝 Key: {key_sig}")
        print(f"📝 Seed bars: {seed_bars}")
        print(f"📝 Target bars: {target_bars}")
        print(f"📝 Mode: {'Free' if self.free_generation else 'Template-guided'}")
        print(f"📝 Sampling: {self.sampling_method}", end='')
        if self.sampling_method == 'top_k':
            print(f" (k={self.top_k})")
        elif self.sampling_method in ['nucleus', 'top_p']:
            print(f" (p={self.top_p})")
        else:
            print()
        
        # Statistics
        rh_count = 0
        lh_count = 0
        
        # Generation loop
        for step in range(max_length):
            current_tick = self._get_current_tick(generated)
            current_bar = (current_tick // self.TICKS_PER_BAR) + 1
            
            # Stop condition
            if current_bar > target_bars:
                print(f"\n✅ Reached target {target_bars} bars")
                break
            
            # Get model prediction
            with torch.no_grad():
                input_seq = generated[-1024:] if len(generated) > 1024 else generated
                if not input_seq:
                    input_seq = [self.vocab.get('START', 0)]
                
                input_tensor = torch.tensor([input_seq], device=self.device)
                style_ids = torch.zeros(1, dtype=torch.long, device=self.device)
                logits, _ = self.model(input_tensor, style_ids)
                next_logits = logits[0, -1, :]
            
            # Apply constraints based on mode
            if self.free_generation:
                # Pure transformer generation
                temperature = self.base_temperature
                bias = torch.zeros_like(next_logits)
            else:
                # Simplified template guidance with slightly stronger bias for harmony
                bias = self._get_simple_bias(
                    generated, current_tick, current_bar,
                    waypoints, rhythm_waypoints
                )
                # Adjust temperature based on beat position
                temperature = self.base_temperature * 0.9 if self._is_strong_beat(current_tick) else self.base_temperature * 1.1
            
            # Sample next token with advanced sampling
            biased_logits = next_logits + bias
            next_token, probs = self._sample_token(biased_logits, temperature)
            
            generated.append(next_token)
            
            # Track statistics
            token_str = self.inv_vocab.get(next_token, '')
            if 'RH_NOTE_ON' in token_str:
                rh_count += 1
                match = re.search(r'(\d+)', token_str)
                if match:
                    self.last_rh_pitch = int(match.group(1))
            elif 'LH_NOTE_ON' in token_str:
                lh_count += 1
                match = re.search(r'(\d+)', token_str)
                if match:
                    self.last_lh_pitch = int(match.group(1))
            
            # Progress
            if step % 100 == 0 and step > 0:
                total = rh_count + lh_count
                if total > 0:
                    avg_entropy = sum(self.entropy_values[-100:]) / min(100, len(self.entropy_values))
                    print(f"📊 Step {step}: Bar {current_bar}, RH/LH: {rh_count/total:.1%}/{lh_count/total:.1%}, Entropy: {avg_entropy:.2f}")
            
            # End token
            if 'END' in token_str:
                print(f"🛑 Model generated END at bar {current_bar}")
                break
        
        # Print final entropy statistics
        if self.entropy_values:
            avg_entropy = sum(self.entropy_values) / len(self.entropy_values)
            print(f"\n📈 Average entropy: {avg_entropy:.2f} (diversity indicator)")
        
        return generated
    
    def _get_simple_bias(self, generated: List[int], current_tick: int,
                         current_bar: int, waypoints: Dict,
                         rhythm_waypoints: Dict) -> torch.Tensor:
        """Simplified bias with adjustable strength for harmony guidance"""
        bias = torch.zeros(len(self.vocab), device=self.device)
        
        # Get last token
        last_token = self.inv_vocab.get(generated[-1], '') if generated else ''
        
        # After TIME_SHIFT, gentle note encouragement
        if 'TIME_SHIFT' in last_token:
            # 50/50 RH/LH preference
            if random.random() > 0.5:
                # RH - slightly stronger chord tone preference for Beethoven character
                chord_tones = self._get_chord_tones(waypoints, current_bar)
                for tone in chord_tones:
                    for pitch in range(self.RH_MIN, self.RH_MAX):
                        if pitch % 12 == tone:
                            token = f"RH_NOTE_ON_{pitch}"
                            if token in self.vocab:
                                # Stronger on strong beats for harmonic clarity
                                if self._is_strong_beat(current_tick):
                                    bias[self.vocab[token]] += 4.0 * self.bias_strength
                                else:
                                    bias[self.vocab[token]] += 2.5 * self.bias_strength
                
                # Also allow non-chord tones with lower bias
                for pitch in range(self.RH_MIN, self.RH_MAX):
                    token = f"RH_NOTE_ON_{pitch}"
                    if token in self.vocab and pitch % 12 not in chord_tones:
                        bias[self.vocab[token]] += 1.0 * self.bias_strength  # Allow passing tones
            else:
                # LH - gentle bass preference with root emphasis
                root = self._get_chord_tones(waypoints, current_bar)[0] if waypoints else 0
                for pitch in range(self.LH_MIN, self.LH_MAX):
                    token = f"LH_NOTE_ON_{pitch}"
                    if token in self.vocab:
                        if pitch % 12 == root:
                            bias[self.vocab[token]] += 3.0 * self.bias_strength  # Slight root preference
                        else:
                            bias[self.vocab[token]] += 2.0 * self.bias_strength
        
        # After NOTE_ON/OFF, encourage TIME_SHIFT
        elif 'NOTE_ON' in last_token or 'NOTE_OFF' in last_token:
            self.recent_note_count += 1
            
            # Get rhythm preference
            rhythm = rhythm_waypoints.get(str(current_bar), 'quarter')
            durations = self._get_rhythm_durations(rhythm)
            
            for dur in durations:
                token = f"TIME_SHIFT_{dur}"
                if token in self.vocab:
                    bias[self.vocab[token]] += 3.0 * self.bias_strength
            
            # After 8 notes, stronger TIME_SHIFT
            if self.recent_note_count >= 8:
                for token, idx in self.vocab.items():
                    if 'TIME_SHIFT' in token:
                        bias[idx] += 5.0 * self.bias_strength
                self.recent_note_count = 0
        
        return bias
    
    def _get_chord_tones(self, waypoints: Dict, bar: int) -> List[int]:
        """Get chord tones for current bar with extended chord types"""
        harmony = waypoints.get(str(bar), 'I')
        if isinstance(harmony, dict):
            harmony = harmony.get('1', 'I')
        
        # Get root interval
        root = 0
        for roman, interval in self.ROMAN_TO_INTERVAL.items():
            if roman in harmony:
                root = interval
                break
        
        # Detect chord type with more variations
        if 'maj7' in harmony.lower():
            structure = self.CHORD_STRUCTURES['maj7']
        elif 'min7' in harmony.lower() or 'm7' in harmony:
            structure = self.CHORD_STRUCTURES['min7']
        elif 'half' in harmony.lower() or 'ø' in harmony:
            structure = self.CHORD_STRUCTURES['half-dim']
        elif 'aug' in harmony.lower() or '+' in harmony:
            structure = self.CHORD_STRUCTURES['aug']
        elif '7' in harmony:
            structure = self.CHORD_STRUCTURES['dom7']
        elif '°' in harmony or 'dim' in harmony.lower():
            structure = self.CHORD_STRUCTURES['dim']
        elif harmony[0].islower():
            structure = self.CHORD_STRUCTURES['minor']
        else:
            structure = self.CHORD_STRUCTURES['major']
        
        return [(root + tone) % 12 for tone in structure]
    
    def _get_rhythm_durations(self, rhythm: str) -> List[int]:
        """Map rhythm to durations"""
        rhythm_map = {
            'sixteenth': [120],
            'eighth': [240],
            'quarter': [480],
            'half': [960],
            'whole': [1920],
            'dotted': [720],
            'long': [960, 1920],
            'mixed': [240, 480],  # Mix of eighth and quarter
            'syncopated': [360, 240, 360],  # Syncopated pattern
        }
        
        if isinstance(rhythm, dict):
            rhythm = 'quarter'
        
        return rhythm_map.get(rhythm, [480])
    
    def _is_strong_beat(self, tick: int) -> bool:
        """Check if strong beat position"""
        beat_position = tick % self.TICKS_PER_BEAT
        beat_in_bar = (tick % self.TICKS_PER_BAR) // self.TICKS_PER_BEAT
        return beat_position == 0 and beat_in_bar in [0, 2]
    
    def _get_current_tick(self, sequence: List[int]) -> int:
        """Calculate current tick"""
        tick = 0
        for token_id in sequence:
            token = self.inv_vocab.get(token_id, '')
            if 'TIME_SHIFT' in token:
                match = re.search(r'(\d+)', token)
                if match:
                    tick += int(match.group(1))
        return tick
    
    def _estimate_bars(self, sequence: List[int]) -> int:
        """Estimate bars in sequence"""
        tick = self._get_current_tick(sequence)
        return (tick // self.TICKS_PER_BAR) + 1 if tick > 0 else 0
    
    def _parse_key_root(self, key_sig: str) -> int:
        """Parse key signature to MIDI pitch"""
        key_map = {
            'C': 60, 'D': 62, 'E': 64, 'F': 65, 'G': 67, 'A': 69, 'B': 71,
            'Cb': 59, 'Db': 61, 'Eb': 63, 'Gb': 66, 'Ab': 68, 'Bb': 70,
            'C#': 61, 'D#': 63, 'F#': 66, 'G#': 68, 'A#': 70,
        }
        
        # Extract root
        root = re.match(r'^[A-G][b#]?', key_sig)
        if root:
            return key_map.get(root.group(), 60)
        return 60


def tokens_to_midi_enhanced(sequence: List[int], inv_vocab: Dict[int, str],
                            output_path: str, tempo: int = 100,
                            skip_seed_length: int = 0):
    """Convert tokens to MIDI with better instrument settings"""
    
    # Skip seed if requested
    if skip_seed_length > 0:
        sequence = sequence[skip_seed_length:]
    
    midi = MIDIFile(2)  # 2 tracks for RH/LH
    
    # Set tempo and program (piano)
    midi.addTempo(0, 0, tempo)
    midi.addTempo(1, 0, tempo)
    midi.addProgramChange(0, 0, 0, 0)  # Piano for RH
    midi.addProgramChange(1, 0, 0, 0)  # Piano for LH
    
    current_tick = 0
    rh_notes = {}
    lh_notes = {}
    
    # Track statistics
    note_count = 0
    max_polyphony_rh = 0
    max_polyphony_lh = 0
    
    for token_id in sequence:
        token = inv_vocab.get(token_id, '')
        
        if 'TIME_SHIFT' in token:
            match = re.search(r'(\d+)', token)
            if match:
                current_tick += int(match.group(1))
                # Track polyphony
                max_polyphony_rh = max(max_polyphony_rh, len(rh_notes))
                max_polyphony_lh = max(max_polyphony_lh, len(lh_notes))
        
        elif 'RH_NOTE_ON' in token:
            match = re.search(r'(\d+)', token)
            if match:
                pitch = int(match.group(1))
                rh_notes[pitch] = current_tick
                note_count += 1
        
        elif 'RH_NOTE_OFF' in token:
            match = re.search(r'(\d+)', token)
            if match:
                pitch = int(match.group(1))
                if pitch in rh_notes:
                    start = rh_notes.pop(pitch)
                    duration = current_tick - start
                    if duration > 0:
                        # Dynamic velocity based on beat position
                        velocity = 85 if (start % 1920) == 0 else 80
                        midi.addNote(0, 0, pitch, start/480, duration/480, velocity)
        
        elif 'LH_NOTE_ON' in token:
            match = re.search(r'(\d+)', token)
            if match:
                pitch = int(match.group(1))
                lh_notes[pitch] = current_tick
                note_count += 1
        
        elif 'LH_NOTE_OFF' in token:
            match = re.search(r'(\d+)', token)
            if match:
                pitch = int(match.group(1))
                if pitch in lh_notes:
                    start = lh_notes.pop(pitch)
                    duration = current_tick - start
                    if duration > 0:
                        # Slightly softer LH
                        velocity = 70 if (start % 1920) == 0 else 65
                        midi.addNote(1, 0, pitch, start/480, duration/480, velocity)
    
    # Close remaining notes with appropriate durations
    for pitch, start in rh_notes.items():
        duration = max(240, current_tick - start)  # Min eighth note
        midi.addNote(0, 0, pitch, start/480, duration/480, 70)
    
    for pitch, start in lh_notes.items():
        duration = max(480, current_tick - start)  # Min quarter note for bass
        midi.addNote(1, 0, pitch, start/480, duration/480, 65)
    
    # Save MIDI
    Path(output_path).parent.mkdir(exist_ok=True)
    with open(output_path, 'wb') as f:
        midi.writeFile(f)
    
    print(f"\n✅ Saved MIDI to {output_path}")
    
    # Enhanced statistics
    total_rh = len([t for t in sequence if 'RH_NOTE_ON' in inv_vocab.get(t, '')])
    total_lh = len([t for t in sequence if 'LH_NOTE_ON' in inv_vocab.get(t, '')])
    
    print(f"\n📊 MIDI Statistics:")
    if total_rh + total_lh > 0:
        print(f"  • RH/LH ratio: {total_rh/(total_rh+total_lh):.1%}/{total_lh/(total_rh+total_lh):.1%}")
    print(f"  • Total notes: {note_count}")
    print(f"  • Max polyphony: RH={max_polyphony_rh}, LH={max_polyphony_lh}")
    
    duration_bars = (current_tick / 1920) if current_tick > 0 else 0
    print(f"  • Duration: {duration_bars:.1f} bars ({current_tick/480:.1f} beats)")


def load_model(checkpoint_path: str, device: torch.device):
    """Load trained model"""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    try:
        from model_ra import create_model
        model = create_model(
            vocab_size=len(checkpoint['vocab']),
            n_styles=10,
            model_size=checkpoint['config'].get('model_size', 'base')
        ).to(device)
    except ImportError:
        print("⚠️ Using fallback model")
        import torch.nn as nn
        
        class SimpleTransformer(nn.Module):
            def __init__(self, vocab_size):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, 512)
                self.transformer = nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(512, 8, 2048, batch_first=True),
                    num_layers=6
                )
                self.output = nn.Linear(512, vocab_size)
            
            def forward(self, x, style_ids):
                x = self.embedding(x)
                x = self.transformer(x)
                return self.output(x), None
        
        model = SimpleTransformer(len(checkpoint['vocab'])).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model = model.to(torch.bfloat16)
    model.eval()
    
    return model, checkpoint['vocab'], checkpoint['inv_vocab']


def main():
    """Main entry point with enhanced options"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Beethoven Music Generator - Enhanced Version',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Template-guided generation
  python beethoven_gen.py --checkpoint model.pt --template template.json --output sonata.mid
  
  # Free generation with template seed
  python beethoven_gen.py --checkpoint model.pt --template template.json --free-generation
  
  # Pure free generation
  python beethoven_gen.py --checkpoint model.pt --free-generation --output free.mid
  
  # With advanced sampling
  python beethoven_gen.py --checkpoint model.pt --sampling nucleus --top-p 0.95
        """
    )
    
    parser.add_argument('--template', help='Template JSON file')
    parser.add_argument('--checkpoint', required=True, help='Model checkpoint')
    parser.add_argument('--output', default='output.mid', help='Output MIDI')
    parser.add_argument('--max-length', type=int, default=768, help='Max tokens')
    parser.add_argument('--tempo', type=int, default=100, help='Tempo BPM')
    parser.add_argument('--random-seed', type=int, help='Random seed for reproducibility')
    parser.add_argument('--free-generation', action='store_true',
                        help='Free generation mode - minimal constraints')
    parser.add_argument('--exclude-seed', action='store_true',
                        help='Exclude seed from MIDI output')
    
    # Sampling options
    parser.add_argument('--sampling', choices=['multinomial', 'argmax', 'top_k', 'nucleus', 'top_p'],
                        default='multinomial', help='Sampling method')
    parser.add_argument('--top-k', type=int, default=50,
                        help='Top-k value for top_k sampling')
    parser.add_argument('--top-p', type=float, default=0.95,
                        help='Top-p value for nucleus sampling')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Base temperature for sampling (0.1-2.0 recommended)')
    parser.add_argument('--bias-strength', type=float, default=1.0,
                        help='Strength of harmonic bias (0.5=weaker, 2.0=stronger)')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")
    
    # Load model
    print(f"📦 Loading model from {args.checkpoint}...")
    model, vocab, inv_vocab = load_model(args.checkpoint, device)
    print(f"  ✓ Vocabulary size: {len(vocab)}")
    
    # Load template or create empty
    if args.template:
        print(f"📄 Loading template from {args.template}...")
        try:
            with open(args.template, 'r') as f:
                template = json.load(f)
            print(f"  ✓ Template loaded")
            
            # Validate template structure
            if 'waypoints' in template:
                for key in template['waypoints'].keys():
                    try:
                        int(key)  # Check if bar numbers are valid
                    except ValueError:
                        print(f"  ⚠️ Warning: Invalid bar number '{key}' in waypoints")
        except json.JSONDecodeError as e:
            print(f"  ❌ Error: Invalid JSON in template file: {e}")
            return
        except FileNotFoundError:
            print(f"  ❌ Error: Template file not found: {args.template}")
            return
    else:
        print("📄 No template - full free generation")
        template = {
            'seed_tokens': [],
            'waypoints': {},
            'rhythm_waypoints': {},
            'metadata': {'key_signature': 'C'}
        }
    
    # Create generator with sampling options
    generator = BeethovenGeneratorRefined(
        model, vocab, inv_vocab, device,
        args.random_seed, args.free_generation,
        args.sampling, args.top_k, args.top_p,
        args.temperature, args.bias_strength
    )
    
    print("\n" + "="*60)
    print("🎼 BEETHOVEN GENERATOR - ENHANCED VERSION")
    print("="*60)
    
    # Generate
    print("\n🎵 Generating music...")
    start_time = time.time()
    sequence = generator.generate(template, args.max_length)
    generation_time = time.time() - start_time
    
    # Convert to MIDI
    print("\n💾 Converting to MIDI...")
    skip_seed = 0
    if args.exclude_seed and args.template:
        skip_seed = len(template.get('seed_tokens', []))
        if skip_seed > 0:
            print(f"  • Excluding {skip_seed} seed tokens")
    
    tokens_to_midi_enhanced(sequence, inv_vocab, args.output, args.tempo, skip_seed)
    
    # Final statistics
    print("\n" + "="*60)
    print("📊 GENERATION COMPLETE")
    print("="*60)
    print(f"  • Total tokens: {len(sequence)}")
    print(f"  • Generation time: {generation_time:.1f}s")
    print(f"  • Tokens/second: {len(sequence)/generation_time:.1f}")
    
    total_ticks = 0
    for t in sequence:
        token = inv_vocab.get(t, '')
        if 'TIME_SHIFT' in token:
            match = re.search(r'(\d+)', token)
            if match:
                total_ticks += int(match.group(1))
    
    if total_ticks > 0:
        duration_bars = total_ticks / 1920
        duration_secs = (total_ticks / 480) * (60 / args.tempo)
        print(f"  • Musical duration: {duration_bars:.1f} bars ({duration_secs:.1f}s @ {args.tempo} BPM)")
    
    print(f"\n✅ Output saved to: {args.output}")
    print("="*60)


if __name__ == "__main__":
    main()