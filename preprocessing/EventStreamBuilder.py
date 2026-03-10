"""
EventStreamBuilder.py v12.0 - FINAL PRODUCTION VERSION

최종 수정:
- Completely removed automatic Tick detection logic
- Directly use start_tick_abs, end_tick_abs (100% trust)
- Maximized stability through code simplification
- Kept all edge cases while removing unnecessary complexity
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, NamedTuple, Set
import numpy as np
import torch
from tqdm import tqdm
from collections import defaultdict, Counter
from datetime import datetime
import midiutil
from midiutil import MIDIFile

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class NoteEvent(NamedTuple):
    """Pure note event for event stream."""
    absolute_tick: int
    event_type: str
    hand: str
    pitch: int
    velocity: int
    note_id: Optional[int] = None  # For tracking overlapping notes


class StructuralMarker(NamedTuple):
    """Structural marker for FSM navigation."""
    absolute_tick: int
    marker_type: str


class EventStreamBuilder:
    """
    FINAL production-ready event stream builder.
    
    v12.0: Simplified and robust - trusts JSON absolute ticks completely
    """
    
    # Constants
    TICKS_PER_QUARTER = 480
    MIN_MIDI_PITCH = 21  # A0
    MAX_MIDI_PITCH = 108  # C8
    MAX_TIME_SHIFT = 1000
    MAX_SEQUENCE_LENGTH = 4096
    SEQUENCE_BUFFER = 200
    
    # Known cadence types
    KNOWN_CADENCE_TYPES = ['HC', 'PAC', 'IAC', 'DC', 'PC']
    
    # Time signature word mappings
    TIME_SIG_WORDS = {
        'one': '1', 'two': '2', 'three': '3', 'four': '4',
        'five': '5', 'six': '6', 'seven': '7', 'eight': '8',
        'nine': '9', 'ten': '10', 'eleven': '11', 'twelve': '12',
        'thirteen': '13', 'fourteen': '14', 'fifteen': '15', 'sixteen': '16'
    }
    
    def __init__(self, json_dir: str, output_dir: str = "output"):
        self.json_dir = Path(json_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if not self.json_dir.exists():
            raise FileNotFoundError(f"JSON directory not found: {json_dir}")
        
        self.json_files = list(self.json_dir.glob("*.json"))
        logger.info(f"Found {len(self.json_files)} JSON files")
        
        # Dynamic cadence types found in data
        self.found_cadence_types = set(self.KNOWN_CADENCE_TYPES)
        
        self.vocab = {}
        self.inv_vocab = {}
        self._scan_for_cadence_types()
        self._create_vocabulary()
        
        self.stats = defaultdict(int)
        
        # Track overlapping notes
        self.note_id_counter = 0
    
    def _scan_for_cadence_types(self) -> None:
        """Scan JSON files for all cadence types used."""
        logger.info("Scanning for cadence types in JSON files...")
        
        for json_path in self.json_files[:10]:  # Sample first 10 files
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                
                for measure in data.get('measures', []):
                    for event in measure.get('events', []):
                        cadence_type = event.get('cadence_type')
                        if cadence_type:
                            self.found_cadence_types.add(cadence_type.upper())
            except Exception as e:
                logger.warning(f"Error scanning {json_path}: {e}")
        
        logger.info(f"Found cadence types: {sorted(self.found_cadence_types)}")
    
    def _create_vocabulary(self) -> None:
        """Create comprehensive vocabulary."""
        logger.info("Creating vocabulary...")
        
        tokens = []
        
        # Special tokens
        tokens.extend(['<PAD>', '<SOS>', '<EOS>', '<UNK>'])
        
        # Structural markers
        tokens.extend([
            '<PHRASE_START>', '<PHRASE_END>',
            '<SECTION_A>', '<SECTION_B>', '<SECTION_A_PRIME>',
            '<SECTION_INTRO>', '<SECTION_CODA>', '<SECTION_DEVELOPMENT>',
            '<MEASURE_BAR>',
            '<CADENCE_UNKNOWN>'
        ])
        
        # All found cadence types
        for cadence in sorted(self.found_cadence_types):
            tokens.append(f'<CADENCE_{cadence}>')
        
        # Note tokens with hand separation
        for pitch in range(self.MIN_MIDI_PITCH, self.MAX_MIDI_PITCH + 1):
            tokens.append(f'RH_NOTE_ON_{pitch}')
            tokens.append(f'LH_NOTE_ON_{pitch}')
            tokens.append(f'RH_NOTE_OFF_{pitch}')
            tokens.append(f'LH_NOTE_OFF_{pitch}')
        
        # Full velocity range
        for velocity in range(128):
            tokens.append(f'VELOCITY_{velocity}')
        
        # Time shift tokens
        for shift in range(1, self.MAX_TIME_SHIFT + 1):
            tokens.append(f'TIME_SHIFT_{shift}')
        
        self.vocab = {token: idx for idx, token in enumerate(tokens)}
        self.inv_vocab = {idx: token for idx, token in enumerate(tokens)}
        
        logger.info(f"Vocabulary created with {len(self.vocab)} tokens")
    
    def _normalize_time_signature(self, time_sig_str: str) -> Tuple[int, int]:
        """Normalize time signature string."""
        if not time_sig_str:
            return 4, 4
        
        time_sig_str = str(time_sig_str).strip()
        
        # Handle music21 object
        music21_match = re.search(r'<music21\.meter\.TimeSignature\s+(\d+/\d+)>', time_sig_str)
        if music21_match:
            parts = music21_match.group(1).split('/')
            return int(parts[0]), int(parts[1])
        
        # Convert to lowercase and replace words
        time_sig_str = time_sig_str.lower()
        for word, digit in self.TIME_SIG_WORDS.items():
            time_sig_str = time_sig_str.replace(word, digit)
        
        # Remove prefixes
        time_sig_str = re.sub(r'time\s*signature\s*', '', time_sig_str)
        
        # Extract pattern
        direct_match = re.search(r'(\d+)/(\d+)', time_sig_str)
        if direct_match:
            return int(direct_match.group(1)), int(direct_match.group(2))
        
        return 4, 4
    
    def _calculate_measure_ticks(self, numerator: int, denominator: int) -> int:
        """Calculate ticks per measure."""
        if denominator == 0:
            return 4 * self.TICKS_PER_QUARTER
        
        ticks = int(numerator * (4.0 / denominator) * self.TICKS_PER_QUARTER)
        return ticks
    
    def _get_measure_starts(self, json_data: Dict) -> List[int]:
        """
        Calculate measure start positions for MEASURE_BAR markers only.
        NOT used for note timing - notes use their own absolute ticks.
        """
        measures = json_data.get('measures', [])
        measure_starts = [0]
        current_tick = 0
        
        # Get global time signature
        global_time_sig = None
        if 'metadata' in json_data:
            global_time_sig = json_data['metadata'].get('time_signature')
        
        if not global_time_sig and measures:
            global_time_sig = measures[0].get('time_signature', '4/4')
        
        global_num, global_den = self._normalize_time_signature(global_time_sig or '4/4')
        
        for i, measure in enumerate(measures[:-1]):
            # Try per-measure time signature
            measure_time_sig = measure.get('time_signature')
            
            if measure_time_sig:
                num, den = self._normalize_time_signature(measure_time_sig)
            else:
                num, den = global_num, global_den
            
            measure_duration = self._calculate_measure_ticks(num, den)
            current_tick += measure_duration
            measure_starts.append(current_tick)
        
        return measure_starts
    
    def _extract_events_from_json(self, json_path: Path) -> Tuple[List[NoteEvent], List[StructuralMarker]]:
        """
        Extract events using ONLY absolute tick values from JSON.
        SIMPLIFIED: No overlap handling for cleaner 1:1 ratio
        """
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        events = []
        structural_markers = []
        
        measures = data.get('measures', [])
        measure_starts = self._get_measure_starts(data)
        
        for measure_idx, measure in enumerate(measures):
            # Add measure boundary marker
            if measure_idx > 0 and measure_idx < len(measure_starts):
                events.append(NoteEvent(
                    absolute_tick=measure_starts[measure_idx],
                    event_type='MEASURE_BAR',
                    hand='',
                    pitch=0,
                    velocity=0,
                    note_id=None
                ))
            
            for event_data in measure.get('events', []):
                event_tick = event_data.get('tick', 0)
                
                # Phrase boundaries
                if event_data.get('is_phrase_boundary', False):
                    marker_type = 'PHRASE_START' if event_tick % self.TICKS_PER_QUARTER < self.TICKS_PER_QUARTER // 2 else 'PHRASE_END'
                    structural_markers.append(StructuralMarker(
                        absolute_tick=event_tick,
                        marker_type=marker_type
                    ))
                    self.stats['phrase_boundaries'] += 1
                
                # Cadences
                cadence_type = event_data.get('cadence_type')
                if cadence_type:
                    cadence_upper = cadence_type.upper()
                    cadence_token = f'<CADENCE_{cadence_upper}>'
                    
                    if cadence_token in self.vocab:
                        marker_type = f'CADENCE_{cadence_upper}'
                        self.stats[f'cadence_{cadence_type.lower()}'] += 1
                    else:
                        marker_type = 'CADENCE_UNKNOWN'
                        self.stats['cadence_unknown'] += 1
                    
                    structural_markers.append(StructuralMarker(
                        absolute_tick=event_tick,
                        marker_type=marker_type
                    ))
                
                # Process right hand notes - SIMPLIFIED (NO OVERLAP HANDLING)
                if event_data.get('rh_action') == 'NOTE_ON':
                    for note in event_data.get('rh_notes', []):
                        start_tick = note.get('start_tick_abs', event_tick)
                        end_tick = note.get('end_tick_abs', start_tick + self.TICKS_PER_QUARTER)
                        velocity = note.get('velocity', 80)
                        pitch = note.get('pitch', 60)
                        
                        if self.MIN_MIDI_PITCH <= pitch <= self.MAX_MIDI_PITCH:
                            # Simple NOTE_ON
                            events.append(NoteEvent(
                                absolute_tick=start_tick,
                                event_type='NOTE_ON',
                                hand='RH',
                                pitch=pitch,
                                velocity=velocity,
                                note_id=None
                            ))
                            
                            # Simple NOTE_OFF
                            events.append(NoteEvent(
                                absolute_tick=end_tick,
                                event_type='NOTE_OFF',
                                hand='RH',
                                pitch=pitch,
                                velocity=0,
                                note_id=None
                            ))
                
                # Process left hand notes - SIMPLIFIED (NO OVERLAP HANDLING)
                if event_data.get('lh_action') == 'NOTE_ON':
                    for note in event_data.get('lh_notes', []):
                        start_tick = note.get('start_tick_abs', event_tick)
                        end_tick = note.get('end_tick_abs', start_tick + self.TICKS_PER_QUARTER)
                        velocity = note.get('velocity', 80)
                        pitch = note.get('pitch', 60)
                        
                        if self.MIN_MIDI_PITCH <= pitch <= self.MAX_MIDI_PITCH:
                            # Simple NOTE_ON
                            events.append(NoteEvent(
                                absolute_tick=start_tick,
                                event_type='NOTE_ON',
                                hand='LH',
                                pitch=pitch,
                                velocity=velocity,
                                note_id=None
                            ))
                            
                            # Simple NOTE_OFF
                            events.append(NoteEvent(
                                absolute_tick=end_tick,
                                event_type='NOTE_OFF',
                                hand='LH',
                                pitch=pitch,
                                velocity=0,
                                note_id=None
                            ))
        
        # Sort events
        events.sort(key=lambda e: (e.absolute_tick, 0 if e.event_type == 'NOTE_ON' else 1))
        
        return events, structural_markers
    
    def _merge_structural_markers(self, events: List[NoteEvent], 
                                 markers: List[StructuralMarker]) -> List[Any]:
        """Merge note events and structural markers."""
        merged = []
        event_idx = 0
        marker_idx = 0
        
        markers = sorted(markers, key=lambda m: m.absolute_tick)
        
        while event_idx < len(events) or marker_idx < len(markers):
            event_tick = events[event_idx].absolute_tick if event_idx < len(events) else float('inf')
            marker_tick = markers[marker_idx].absolute_tick if marker_idx < len(markers) else float('inf')
            
            if marker_tick < event_tick:
                merged.append(markers[marker_idx])
                marker_idx += 1
            else:
                merged.append(events[event_idx])
                event_idx += 1
        
        return merged
    
    def _events_to_tokens(self, merged_events: List[Any]) -> List[str]:
        """Convert events to tokens with relative time shifts."""
        tokens = []
        last_tick = 0
        
        for item in merged_events:
            if isinstance(item, StructuralMarker):
                if item.absolute_tick > last_tick:
                    time_delta = item.absolute_tick - last_tick
                    while time_delta > 0:
                        shift = min(time_delta, self.MAX_TIME_SHIFT)
                        tokens.append(f'TIME_SHIFT_{shift}')
                        time_delta -= shift
                
                tokens.append(f'<{item.marker_type}>')
                last_tick = item.absolute_tick
                
            elif isinstance(item, NoteEvent):
                if item.absolute_tick > last_tick:
                    time_delta = item.absolute_tick - last_tick
                    while time_delta > 0:
                        shift = min(time_delta, self.MAX_TIME_SHIFT)
                        tokens.append(f'TIME_SHIFT_{shift}')
                        time_delta -= shift
                
                if item.event_type == 'NOTE_ON':
                    tokens.append(f'VELOCITY_{item.velocity}')
                    tokens.append(f'{item.hand}_NOTE_ON_{item.pitch}')
                elif item.event_type == 'NOTE_OFF':
                    tokens.append(f'{item.hand}_NOTE_OFF_{item.pitch}')
                elif item.event_type == 'MEASURE_BAR':
                    tokens.append('<MEASURE_BAR>')
                
                last_tick = item.absolute_tick
        
        return tokens
    
    def _split_long_sequence_with_active_notes(self, tokens: List[str], 
                                              indices: List[int]) -> List[List[int]]:
        """
        Split sequences with COMPLETE polyphony support.
        Now tracks multiple overlapping notes per pitch.
        """
        if len(indices) <= self.MAX_SEQUENCE_LENGTH:
            return [indices]
        
        logger.info(f"Splitting sequence of {len(indices)} tokens")
        sequences = []
        remaining_indices = indices.copy()
        
        # Track multiple active notes per pitch - LIST of (velocity, note_id)
        current_active_notes = {'RH': defaultdict(list), 'LH': defaultdict(list)}
        
        while len(remaining_indices) > self.MAX_SEQUENCE_LENGTH:
            best_split_point = -1
            search_start = max(self.MAX_SEQUENCE_LENGTH - self.SEQUENCE_BUFFER, 1)
            search_end = min(self.MAX_SEQUENCE_LENGTH, len(remaining_indices))
            
            # Deep copy for scanning - preserve list structure
            temp_active = {
                'RH': defaultdict(list, {k: v.copy() for k, v in current_active_notes['RH'].items()}),
                'LH': defaultdict(list, {k: v.copy() for k, v in current_active_notes['LH'].items()})
            }
            pending_velocity = None
            note_id_counter = 0  # Track note IDs for overlap management
            
            # Scan through to find good split point
            for i in range(1, search_end):
                if i < len(remaining_indices):
                    token_str = self.inv_vocab.get(remaining_indices[i], "")
                    
                    if token_str.startswith('VELOCITY_'):
                        pending_velocity = int(token_str.split('_')[1])
                    
                    elif 'NOTE_ON' in token_str:
                        parts = token_str.split('_')
                        if len(parts) >= 4:
                            hand = parts[0]
                            pitch = int(parts[3])
                            
                            if pending_velocity is not None:
                                velocity = pending_velocity
                                pending_velocity = None
                            else:
                                logger.debug(f"NOTE_ON without VELOCITY at position {i}, using default 80")
                                velocity = 80
                            
                            # Add to list of active notes for this pitch
                            temp_active[hand][pitch].append((velocity, note_id_counter))
                            note_id_counter += 1
                    
                    elif 'NOTE_OFF' in token_str:
                        parts = token_str.split('_')
                        if len(parts) >= 4:
                            hand = parts[0]
                            pitch = int(parts[3])
                            
                            # Remove oldest note from the list (FIFO for overlaps)
                            if temp_active[hand][pitch]:
                                temp_active[hand][pitch].pop(0)
                                if not temp_active[hand][pitch]:
                                    del temp_active[hand][pitch]
                    
                    # Check for clean split points
                    if i >= search_start:
                        if token_str in ['<MEASURE_BAR>', '<PHRASE_END>', '<PHRASE_START>']:
                            best_split_point = i
                            # Deep copy the current state
                            current_active_notes = {
                                'RH': defaultdict(list, {k: v.copy() for k, v in temp_active['RH'].items()}),
                                'LH': defaultdict(list, {k: v.copy() for k, v in temp_active['LH'].items()})
                            }
                            break
            
            # Force split if no ideal point found
            if best_split_point <= 0:
                best_split_point = min(self.MAX_SEQUENCE_LENGTH - self.SEQUENCE_BUFFER, 
                                      len(remaining_indices) - 1)
                
                # Rescan to get accurate state at split point
                temp_active = {'RH': defaultdict(list), 'LH': defaultdict(list)}
                pending_velocity = None
                note_id_counter = 0
                
                for i in range(1, best_split_point):
                    token_str = self.inv_vocab.get(remaining_indices[i], "")
                    
                    if token_str.startswith('VELOCITY_'):
                        pending_velocity = int(token_str.split('_')[1])
                    
                    elif 'NOTE_ON' in token_str:
                        parts = token_str.split('_')
                        if len(parts) >= 4:
                            hand = parts[0]
                            pitch = int(parts[3])
                            velocity = pending_velocity if pending_velocity is not None else 80
                            temp_active[hand][pitch].append((velocity, note_id_counter))
                            note_id_counter += 1
                            pending_velocity = None
                    
                    elif 'NOTE_OFF' in token_str:
                        parts = token_str.split('_')
                        if len(parts) >= 4:
                            hand = parts[0]
                            pitch = int(parts[3])
                            if temp_active[hand][pitch]:
                                temp_active[hand][pitch].pop(0)
                                if not temp_active[hand][pitch]:
                                    del temp_active[hand][pitch]
                
                current_active_notes = temp_active
                logger.warning("No ideal boundary found, forcing split")
            
            # Create chunk
            chunk = remaining_indices[:best_split_point]
            if not chunk or chunk[-1] != self.vocab['<EOS>']:
                chunk.append(self.vocab['<EOS>'])
            sequences.append(chunk)
            
            # Prepare remaining tokens
            remaining_indices = remaining_indices[best_split_point:]
            
            # Create next sequence with ALL active notes restored
            new_sequence_start = [self.vocab['<SOS>']]
            
            # Restore ALL overlapping notes with their specific velocities
            notes_restored = 0
            for hand in ['RH', 'LH']:
                for pitch in sorted(current_active_notes[hand].keys()):
                    # Restore EACH overlapping note
                    for velocity, note_id in current_active_notes[hand][pitch]:
                        new_sequence_start.append(self.vocab[f'VELOCITY_{velocity}'])
                        new_sequence_start.append(self.vocab[f'{hand}_NOTE_ON_{pitch}'])
                        notes_restored += 1
                        self.stats['split_active_notes_restored'] += 1
            
            if notes_restored > 0:
                logger.debug(f"Restored {notes_restored} active notes (including overlaps)")
            
            # Add the rest
            remaining_indices = new_sequence_start + remaining_indices
            self.stats['sequences_split'] += 1
        
        # Add final chunk
        if remaining_indices:
            if not remaining_indices or remaining_indices[-1] != self.vocab['<EOS>']:
                remaining_indices.append(self.vocab['<EOS>'])
            sequences.append(remaining_indices)
        
        logger.info(f"Split into {len(sequences)} sequences")
        return sequences
    
    def process_file(self, json_path: Path) -> Dict[str, Any]:
        """Process a single JSON file."""
        logger.info(f"Processing {json_path.name}")
        
        self.note_id_counter = 0
        
        events, structural_markers = self._extract_events_from_json(json_path)
        
        self.stats['total_events'] += len(events)
        self.stats['note_on_events'] += sum(1 for e in events if e.event_type == 'NOTE_ON')
        self.stats['note_off_events'] += sum(1 for e in events if e.event_type == 'NOTE_OFF')
        self.stats['structural_markers'] += len(structural_markers)
        
        merged = self._merge_structural_markers(events, structural_markers)
        tokens = self._events_to_tokens(merged)
        tokens = ['<SOS>'] + tokens + ['<EOS>']
        
        indices = []
        for token in tokens:
            if token in self.vocab:
                indices.append(self.vocab[token])
                self.stats['token_counts'] += 1
            else:
                logger.warning(f"Unknown token: {token}")
                indices.append(self.vocab['<UNK>'])
                self.stats['unknown_tokens'] += 1
        
        return {
            'file_name': json_path.name,
            'tokens': tokens,
            'indices': indices,
            'length': len(indices),
            'events': events,
            'markers': structural_markers
        }
    
    def build_dataset(self) -> Dict[str, Any]:
        """Build complete dataset."""
        logger.info("Building FINAL production dataset...")
        
        all_sequences = []
        metadata = []
        
        for json_path in tqdm(self.json_files, desc="Processing files"):
            try:
                result = self.process_file(json_path)
                sequences = self._split_long_sequence_with_active_notes(
                    result['tokens'], result['indices']
                )
                
                for seq_idx, sequence in enumerate(sequences):
                    all_sequences.append(sequence)
                    metadata.append({
                        'source_file': result['file_name'],
                        'sequence_length': len(sequence),
                        'is_split': len(sequences) > 1,
                        'split_index': seq_idx if len(sequences) > 1 else None,
                        'num_structural_markers': len(result['markers'])
                    })
                
                self.stats['files_processed'] += 1
                
            except Exception as e:
                logger.error(f"Error processing {json_path}: {e}")
                self.stats['files_failed'] += 1
        
        if not all_sequences:
            raise ValueError("No sequences generated!")
        
        max_length = max(len(seq) for seq in all_sequences)
        logger.info(f"Max sequence length: {max_length}")
        
        padded_sequences = []
        pad_idx = self.vocab['<PAD>']
        
        for seq in all_sequences:
            if len(seq) < max_length:
                padded = seq + [pad_idx] * (max_length - len(seq))
            else:
                padded = seq[:max_length]
            padded_sequences.append(padded)
        
        sequences_tensor = torch.tensor(padded_sequences, dtype=torch.long)
        
        if self.stats['note_on_events'] > 0:
            note_ratio = self.stats['note_off_events'] / self.stats['note_on_events']
        else:
            note_ratio = 0.0
        
        dataset = {
            'sequences': sequences_tensor,
            'vocab': self.vocab,
            'inv_vocab': self.inv_vocab,
            'metadata': metadata,
            'stats': dict(self.stats),
            'config': {
                'vocab_size': len(self.vocab),
                'max_seq_length': max_length,
                'ticks_per_quarter': self.TICKS_PER_QUARTER,
                'note_on_off_ratio': note_ratio,
                'version': '12.0-FINAL',
                'timestamp': datetime.now().isoformat(),
                'cadence_types_found': sorted(list(self.found_cadence_types))
            }
        }
        
        return dataset
    
    def create_smart_embedding_info(self) -> Dict[str, torch.Tensor]:
        """Create mappings for Smart Embedding."""
        vocab_size = len(self.vocab)
        
        token_to_pitch = torch.zeros(vocab_size, dtype=torch.long)
        token_to_hand = torch.zeros(vocab_size, dtype=torch.long)
        is_note_token = torch.zeros(vocab_size, dtype=torch.bool)
        
        for token_str, token_id in self.vocab.items():
            if 'NOTE_ON' in token_str or 'NOTE_OFF' in token_str:
                parts = token_str.split('_')
                
                if parts[0] == 'RH':
                    token_to_hand[token_id] = 1
                elif parts[0] == 'LH':
                    token_to_hand[token_id] = 2
                
                try:
                    pitch = int(parts[-1])
                    token_to_pitch[token_id] = pitch - 21
                    is_note_token[token_id] = True
                except:
                    pass
        
        return {
            'token_to_pitch': token_to_pitch,
            'token_to_hand': token_to_hand,
            'is_note_token': is_note_token
        }
    
    def save_dataset(self, dataset: Dict[str, Any], output_name: str = "beethoven_event_stream.pt"):
        """Save dataset."""
        output_path = self.output_dir / output_name
        torch.save(dataset, output_path)
        logger.info(f"Dataset saved to {output_path}")
        self.print_statistics(dataset)
    
    def print_statistics(self, dataset: Dict[str, Any]):
        """Print dataset statistics."""
        print("\n" + "=" * 70)
        print("FINAL PRODUCTION DATASET (v12.0)")
        print("=" * 70)
        
        stats = dataset['stats']
        config = dataset['config']
        
        print(f"\nDataset Summary:")
        print(f"  Total sequences: {len(dataset['sequences'])}")
        print(f"  Vocabulary size: {config['vocab_size']}")
        print(f"  Max sequence length: {config['max_seq_length']}")
        print(f"  Files processed: {stats.get('files_processed', 0)}")
        print(f"  Files failed: {stats.get('files_failed', 0)}")
        
        print(f"\nEvent Statistics:")
        print(f"  Total events: {stats.get('total_events', 0):,}")
        print(f"  NOTE_ON events: {stats.get('note_on_events', 0):,}")
        print(f"  NOTE_OFF events: {stats.get('note_off_events', 0):,}")
        print(f"  NOTE ON/OFF ratio: {config.get('note_on_off_ratio', 0):.3f}")
        
        print(f"\nStructural Information:")
        print(f"  Total structural markers: {stats.get('structural_markers', 0)}")
        print(f"  Phrase boundaries: {stats.get('phrase_boundaries', 0)}")
        print(f"  Unknown cadences: {stats.get('cadence_unknown', 0)}")
        
        print(f"\nSequence Splitting:")
        print(f"  Sequences split: {stats.get('sequences_split', 0)}")
        print(f"  Active notes restored: {stats.get('split_active_notes_restored', 0)}")
        
        print("\n" + "=" * 70)
    
    def tokens_to_midi(self, tokens: List[int], output_path: str, tempo: int = 120):
        """Convert tokens to MIDI."""
        midi = MIDIFile(2)
        midi.addTempo(0, 0, tempo)
        midi.addTempo(1, 0, tempo)
        
        active_notes = {'RH': {}, 'LH': {}}
        current_tick = 0
        current_velocity = 80
        
        for token_id in tokens:
            token = self.inv_vocab.get(token_id, '<UNK>')
            
            if token.startswith('<'):
                continue
            
            if token.startswith('TIME_SHIFT_'):
                shift = int(token.split('_')[2])
                current_tick += shift
            elif token.startswith('VELOCITY_'):
                current_velocity = int(token.split('_')[1])
            elif 'NOTE_ON' in token:
                parts = token.split('_')
                hand = parts[0]
                pitch = int(parts[3])
                track = 0 if hand == 'RH' else 1
                active_notes[hand][pitch] = (current_tick, current_velocity)
            elif 'NOTE_OFF' in token:
                parts = token.split('_')
                hand = parts[0]
                pitch = int(parts[3])
                track = 0 if hand == 'RH' else 1
                
                if pitch in active_notes[hand]:
                    start_tick, velocity = active_notes[hand].pop(pitch)
                    duration = (current_tick - start_tick) / self.TICKS_PER_QUARTER
                    if duration > 0:
                        start_time = start_tick / self.TICKS_PER_QUARTER
                        midi.addNote(track, 0, pitch, start_time, duration, velocity)
        
        for hand, notes in active_notes.items():
            track = 0 if hand == 'RH' else 1
            for pitch, (start_tick, velocity) in notes.items():
                duration = (current_tick - start_tick) / self.TICKS_PER_QUARTER
                if duration > 0:
                    start_time = start_tick / self.TICKS_PER_QUARTER
                    midi.addNote(track, 0, pitch, start_time, duration, velocity)
        
        with open(output_path, "wb") as f:
            midi.writeFile(f)
        logger.info(f"MIDI saved to {output_path}")


def main():
    """Main execution."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="FINAL Production Dataset Builder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
FINAL VERSION (v12.0) - Simple, Robust, Production-Ready

✓ Trusts JSON absolute ticks (start_tick_abs, end_tick_abs)
✓ No complex tick detection - just uses the data as-is
✓ All edge cases handled without over-engineering
✓ Ready for immediate production use

Example:
  python EventStreamBuilder.py --json_dir beethoven_ticks_v5 --output_dir output
        """
    )
    
    parser.add_argument('--json_dir', type=str, default='beethoven_ticks_v5',
                       help='Directory containing JSON files')
    parser.add_argument('--output_dir', type=str, default='output',
                       help='Output directory')
    parser.add_argument('--output_name', type=str, default='beethoven_final.pt',
                       help='Output file name')
    parser.add_argument('--demo_midi', action='store_true',
                       help='Generate demo MIDI files')
    
    args = parser.parse_args()
    
    builder = EventStreamBuilder(args.json_dir, args.output_dir)
    dataset = builder.build_dataset()
    
    embedding_info = builder.create_smart_embedding_info()
    dataset['embedding_info'] = embedding_info
    
    builder.save_dataset(dataset, args.output_name)
    
    if args.demo_midi:
        logger.info("Generating demo MIDI files...")
        for i in range(min(3, len(dataset['sequences']))):
            seq = dataset['sequences'][i].tolist()
            seq = [t for t in seq if t != dataset['vocab']['<PAD>']]
            
            output_path = builder.output_dir / f'demo_{i}.mid'
            builder.tokens_to_midi(seq, str(output_path))
    
    logger.info("✅ FINAL dataset ready for production!")
    print(f"\n🚀 SIMPLICITY = RELIABILITY")
    print(f"  ✓ Direct use of start_tick_abs/end_tick_abs")
    print(f"  ✓ No tick format guessing")
    print(f"  ✓ Clean, maintainable code")
    print(f"  ✓ Production-ready!")


if __name__ == "__main__":
    main()