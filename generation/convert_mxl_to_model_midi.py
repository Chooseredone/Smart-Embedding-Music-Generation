import music21
import json
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import argparse
import sys
import torch
from midiutil import MIDIFile
import traceback

# --- Start Copy: tokens_to_midi_enhanced function from beethoven_gen.py ---
# This function must be exactly the same as how the AI model converts tokens to MIDI.
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
    
    # Handle potential non-integer keys in inv_vocab (robustness)
    if inv_vocab and isinstance(next(iter(inv_vocab.keys())), str):
        inv_vocab = {int(k): v for k, v in inv_vocab.items()}

    for token_id in sequence:
        token = inv_vocab.get(token_id, '')
        
        if 'TIME_SHIFT' in token:
            match = re.search(r'(\d+)', token)
            if match:
                current_tick += int(match.group(1))
        
        elif 'RH_NOTE_ON' in token:
            match = re.search(r'(\d+)', token)
            if match:
                pitch = int(match.group(1))
                rh_notes[pitch] = current_tick
        
        elif 'RH_NOTE_OFF' in token:
            match = re.search(r'(\d+)', token)
            if match:
                pitch = int(match.group(1))
                if pitch in rh_notes:
                    start = rh_notes.pop(pitch)
                    duration = current_tick - start
                    if duration > 0:
                        # Dynamic velocity based on beat position (AI model's logic)
                        # 1920 ticks = 1 bar (in 4/4 time at 480 TPB)
                        velocity = 85 if (start % 1920) == 0 else 80
                        # Note: time and duration are in beats (tick/480)
                        midi.addNote(0, 0, pitch, start/480, duration/480, velocity)
        
        elif 'LH_NOTE_ON' in token:
            match = re.search(r'(\d+)', token)
            if match:
                pitch = int(match.group(1))
                lh_notes[pitch] = current_tick
        
        elif 'LH_NOTE_OFF' in token:
            match = re.search(r'(\d+)', token)
            if match:
                pitch = int(match.group(1))
                if pitch in lh_notes:
                    start = lh_notes.pop(pitch)
                    duration = current_tick - start
                    if duration > 0:
                        # Slightly softer LH (AI model's logic)
                        velocity = 70 if (start % 1920) == 0 else 65
                        midi.addNote(1, 0, pitch, start/480, duration/480, velocity)
    
    # Close remaining notes (AI model's logic)
    for pitch, start in rh_notes.items():
        duration = max(240, current_tick - start)  # Min eighth note
        midi.addNote(0, 0, pitch, start/480, duration/480, 70)
    
    for pitch, start in lh_notes.items():
        duration = max(480, current_tick - start)  # Min quarter note for bass
        midi.addNote(1, 0, pitch, start/480, duration/480, 65)
    
    # Save MIDI
    Path(output_path).parent.mkdir(exist_ok=True, parents=True)
    with open(output_path, 'wb') as f:
        midi.writeFile(f)
    
    print(f"\n✅ Saved MIDI to {output_path}")

# --- End Copy ---


class MusicXMLToModelMIDIConverter:
    """
    Converts MusicXML to MIDI using the exact pipeline as the AI model.
    """
    def __init__(self, ticks_per_beat: int = 480, checkpoint_path: Optional[str] = None):
        self.TICKS_PER_BEAT = ticks_per_beat
        self.vocab = {}
        self.inv_vocab = {}
        
        if checkpoint_path:
            self.load_vocab_from_checkpoint(checkpoint_path)
        else:
            print("Error: Checkpoint path is required.")
            sys.exit(1)

    def load_vocab_from_checkpoint(self, checkpoint_path: str):
        try:
            # Load checkpoint on CPU. 
            # Note: PyTorch might warn about security if weights_only=False is used implicitly or explicitly.
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            self.vocab = checkpoint.get('vocab', {})
            self.inv_vocab = checkpoint.get('inv_vocab', {})
            if not self.vocab or not self.inv_vocab:
                print("Fatal Error: Vocab not found in checkpoint.")
                sys.exit(1)
            print(f"Vocabulary loaded successfully (Size: {len(self.vocab)}).")

        except Exception as e:
            print(f"Fatal Error loading checkpoint: {e}.")
            sys.exit(1)


    def _get_time_signature_info(self, score):
        # Robust time signature handling
        try:
            # Look for the first time signature
            ts = score.flat.getElementsByClass(music21.meter.TimeSignature)[0]
            # Use music21's calculated duration in quarter lengths
            beats_per_bar = ts.barDuration.quarterLength
            return ts.numerator, beats_per_bar
        except IndexError:
            print("Warning: Time signature not found. Assuming 4/4.")
            return 4, 4.0

    def parse_and_tokenize(self, xml_path: str, start_bar: Optional[int] = None, end_bar: Optional[int] = None) -> List[int]:
        print(f"Parsing MusicXML: {xml_path}")
        try:
            score = music21.converter.parse(xml_path)
        except Exception as e:
            print(f"Error parsing MusicXML file {xml_path}: {e}")
            return []
        
        # Determine beats per bar for offset calculations
        _, beats_per_bar = self._get_time_signature_info(score)
        ticks_per_bar = self.TICKS_PER_BEAT * beats_per_bar

        events = []

        # Helper function to calculate absolute tick relative to the start of the extraction
        def get_abs_tick(measure_num, offset):
            base_measure = start_bar if start_bar is not None else 1
            # Ensure measure_num is not less than base_measure (handles pickup measures numbered 0)
            if measure_num < base_measure:
                measure_num = base_measure

            measure_offset_ticks = (measure_num - base_measure) * ticks_per_bar
            beat_offset_ticks = int(offset * self.TICKS_PER_BEAT)
            return measure_offset_ticks + beat_offset_ticks

        # Iterate through parts (staves)
        for part_index, part in enumerate(score.parts):
            # Default hand based on staff index (0=RH, 1=LH)
            default_hand = 'RH' if part_index == 0 else 'LH'
            
            print(f"Processing Part {part_index}...")

            # Track the current clef (crucial for hand assignment, mirroring training pipeline)
            current_clef_obj = None

            # Iterate through measures
            for measure in part.getElementsByClass('Measure'):
                measure_num = measure.number
                
                # Skip if outside the desired range
                if start_bar is not None and measure_num < start_bar:
                    continue
                if end_bar is not None and measure_num > end_bar:
                    break
                
                # Determine the current clef context. getContextByClass looks backwards if not found locally.
                current_clef_obj = measure.getContextByClass('Clef')

                # Determine hand based on the current clef
                if isinstance(current_clef_obj, music21.clef.TrebleClef):
                    measure_hand = 'RH'
                elif isinstance(current_clef_obj, music21.clef.BassClef):
                    measure_hand = 'LH'
                else:
                    measure_hand = default_hand 

                # Process notes and chords
                for element in measure.flat.notes: # Includes both Notes and Chords
                    
                    pitches = []
                    if isinstance(element, music21.note.Note):
                        pitches.append(element.pitch)
                    elif isinstance(element, music21.chord.Chord):
                        pitches = element.pitches

                    start_tick = get_abs_tick(measure_num, element.offset)
                    duration_ticks = int(element.duration.quarterLength * self.TICKS_PER_BEAT)
                    end_tick = start_tick + duration_ticks

                    for pitch in pitches:
                        # Hand assignment based on the clef logic
                        hand = measure_hand 

                        events.append({'tick': start_tick, 'type': 'ON', 'hand': hand, 'pitch': pitch.midi})
                        events.append({'tick': end_tick, 'type': 'OFF', 'hand': hand, 'pitch': pitch.midi})

        # Sort events chronologically (Process OFF before ON at the same tick for stability)
        # Sorting key: (Tick, Priority (OFF=0, ON=1), Pitch)
        events.sort(key=lambda x: (x['tick'], x['type']=='ON', x['pitch'])) 

        # Tokenization
        tokens = []
        current_tick = 0

        for event in events:
            # 1. TIME_SHIFT
            if event['tick'] > current_tick:
                shift = event['tick'] - current_tick
                # Split large shifts (Max shift assumed 1920 based on training data)
                while shift > 0:
                    token_shift = min(shift, 1920)
                    token_str = f"TIME_SHIFT_{int(token_shift)}"
                    if token_str in self.vocab:
                        tokens.append(self.vocab[token_str])
                    shift -= token_shift
                current_tick = event['tick']

            # 2. NOTE_ON/OFF
            token_str = f"{event['hand']}_NOTE_{event['type']}_{event['pitch']}"
            if token_str in self.vocab:
                tokens.append(self.vocab[token_str])

        return tokens

    def convert(self, xml_path: str, output_midi_path: str, start_bar: Optional[int] = None, end_bar: Optional[int] = None, tempo: int = 100):
        
        # Step 1 & 2: Parse MXL and Tokenize
        tokens = self.parse_and_tokenize(xml_path, start_bar, end_bar)
        
        if not tokens:
            print("Error: Tokenization resulted in empty sequence.")
            return

        print(f"Generated {len(tokens)} tokens.")

        # Step 3: Render MIDI using the model's renderer
        print("Rendering MIDI using model's pipeline (tokens_to_midi_enhanced)...")
        # This call ensures the velocity and timing rules match the AI generation
        tokens_to_midi_enhanced(tokens, self.inv_vocab, output_midi_path, tempo=tempo)


if __name__ == '__main__':
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Convert MusicXML to MIDI using the AI Model's exact pipeline (MXL -> Tokens -> MIDI Renderer). Ensures fair comparison.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example Usage:
  python convert_mxl_to_model_midi.py input.mxl output_gt.mid --checkpoint path/to/model.pt --start-bar 5 --end-bar 12
        """
    )
    
    parser.add_argument('input_mxl', help='Input MusicXML file path.')
    parser.add_argument('output_midi', help='Output MIDI file path.')
    parser.add_argument('--checkpoint', required=True, help='Path to the model checkpoint (needed for vocabulary).')
    parser.add_argument('--start-bar', type=int, default=None, help='Starting measure number (inclusive).')
    parser.add_argument('--end-bar', type=int, default=None, help='Ending measure number (inclusive).')
    parser.add_argument('--tempo', type=int, default=100, help='Tempo for the output MIDI (BPM).')

    # Handle execution in interactive environments (like notebooks)
    if 'ipykernel' in sys.modules:
        print("Running in interactive mode. Please use the functions directly.")
    else:
        args = parser.parse_args()

        # Initialize the converter
        converter = MusicXMLToModelMIDIConverter(checkpoint_path=args.checkpoint)
        
        # Perform the conversion
        try:
            converter.convert(args.input_mxl, args.output_midi, args.start_bar, args.end_bar, args.tempo)
        except Exception as e:
            print(f"\nAn unexpected error occurred during conversion:")
            traceback.print_exc()