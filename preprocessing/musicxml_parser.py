#!/usr/bin/env python3
"""
musicxml_parser.py v5.1 - Version with all issues fixed

Directly parses MusicXML files of Beethoven's 32 sonatas<br>and converts them into tick-based data for the Lego block system

v5.1 Modifications:
- Handle Time Signature changes (e.g., Op.111)
- Detect and warn about unclosed NOTE_ON events
- Support individual velocity within chords
- Fill harmony information gaps (propagate previous harmony)
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict, field
from collections import defaultdict
import music21 as m21
from tqdm import tqdm
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== Data Structure v5.1 ====================

@dataclass
class Note:
    """Individual note information v5.1 - Includes start/end times"""
    pitch: int  # MIDI number
    velocity: int  # 0-127
    start_tick_abs: int  # Absolute start time
    end_tick_abs: int  # Absolute end time

@dataclass 
class TickEvent:
    """Event for a single tick v5.1"""
    tick: int
    rh_action: str  # 'NOTE_ON', 'HOLD' (NOTE_OFF removed)
    rh_notes: List[Note]
    lh_action: str
    lh_notes: List[Note]
    # Structural Info
    chord_symbol: Optional[str] = None
    is_phrase_boundary: bool = False
    cadence_type: Optional[str] = None

# ==================== HarmonyNormalizer (No changes) ====================

class HarmonyNormalizer:
    """Harmony label normalization v4.0.1 - Optimized processing order + case handling"""
    
    def __init__(self):
        # Basic mappings
        self.simple_mappings = {
            'i': 'i', 'I': 'I', 'ii': 'ii', 'II': 'II',
            'iii': 'iii', 'III': 'III', 'iv': 'iv', 'IV': 'IV',
            'v': 'v', 'V': 'V', 'vi': 'vi', 'VI': 'VI',
            'vii': 'vii', 'VII': 'VII',
            
            # 7th chords
            'V7': 'V7', 'v7': 'v7',
            'I7': 'I7', 'i7': 'i7',
            'II7': 'II7', 'ii7': 'ii7',
            'III7': 'III7', 'iii7': 'iii7',
            'IV7': 'IV7', 'iv7': 'iv7',
            'VI7': 'VI7', 'vi7': 'vi7',
            'VII7': 'VII7', 'vii7': 'vii7',
            
            # Special chords
            'dim': 'vii°', 'aug': 'III+',
            'N': 'N', 'bII': 'N', 'neapolitan': 'N',
            'It+6': 'It+6', 'Fr+6': 'Fr+6', 'Ger+6': 'Ger+6',
        }
        
        # Advanced patterns
        self.advanced_patterns = {
            r'dominant\s+seventh': 'V7',
            r'incomplete\s+dominant[-\s]?seventh': 'V7',
            r'incomplete\s+major': 'I',
            r'incomplete\s+minor': 'i',
            r'neapolitan': 'N',
            r'italian': 'It+6',
            r'french': 'Fr+6', 
            r'german': 'Ger+6',
            r'pedal': 'ped',
            r'perfect\s+fifth': 'P5',
            r'perfect\s+fourth': 'P4',
        }
    
    def clean_chord_label(self, label: str) -> str:
        """
        화음 레이블 정규화 - 모든 정규화를 한 곳에서 처리 (v4.0 최종판)
        가장 복잡한 케이스부터 순서대로 처리하는 것이 중요.
        """
        label = label.strip()

        # Step 1: First, process compound words that have hyphens within the word itself, like 'minor-diminished'
        replacements = {
            "minor-diminished": "minor diminished",
            "major-diminished": "major diminished",
            "half-diminished": "half diminished",
            "minor-augmented": "minor augmented",
            "major-augmented": "major augmented",
            "all-interval": "all interval",
            "chromatic-trimirror": "chromatic trimirror",
            "dominant-seventh": "dominant seventh",
            "major-seventh": "major seventh",
            "minor-seventh": "minor seventh"
        }
        for pattern, replacement in replacements.items():
            # Case-insensitive replacement
            label = re.sub(pattern, replacement, label, flags=re.IGNORECASE)

        # Step 2: Now convert hyphens between note names and modifiers to spaces
        label = re.sub(r'([A-G][#b]?)-([a-zA-Z])', r'\1 \2', label)

        # Step 3: Treat remaining hyphens as flats (b)
        label = label.replace('-', 'b')

        # Step 4: Clean up multiple spaces to single space
        label = re.sub(r'\s+', ' ', label)

        return label
    
    def create_key_safely(self, key_signature):
            """Safely create a music21 Key object"""
            try:
                # Normalize non-standard flat notation returned by music21
                key_signature = str(key_signature).strip()
                
                # Convert '-' to 'b' (D- major → Db major)
                key_signature = key_signature.replace('- ', 'b ')
                key_signature = key_signature.replace('-', 'b')
                
                # Additional normalization (just in case)
                key_signature = key_signature.replace('♭', 'b')
                key_signature = key_signature.replace('♯', '#')
                
                # Case-insensitive matching (accepts both 'a minor' and 'A minor')
                key_pattern = re.match(r'^([A-G][#b]?)\s+(major|minor)$', key_signature.strip(), re.IGNORECASE)
                
                if key_pattern:
                    tonic = key_pattern.group(1).capitalize() # Ensure 'bb' becomes 'Bb'
                    mode = key_pattern.group(2).lower()
                    
                    if mode == 'major':
                        return m21.key.Key(tonic)
                    else:
                        return m21.key.Key(tonic, 'minor')
                else:
                    logger.warning(f"Non-standard key: '{key_signature}' → Replaced with C major")
                    return m21.key.Key('C')
                    
            except Exception as e:
                logger.error(f"Key creation failed: '{key_signature}' - {e}")
                return m21.key.Key('C')
    
    def normalize_with_context(self, chord_label, key_signature, prev_chord=None, next_chord=None):
        """Context-aware harmony normalization (v4.0 simplified)"""
        if not chord_label or chord_label == 'None':
            return {'symbol': 'N.C.', 'tags': []}
        
        if ' above ' in chord_label.lower():
            return {'symbol': 'X_Interval', 'tags': ['interval']}

        cleaned_label = self.clean_chord_label(chord_label)
        
        try:
            # Try music21 analysis if it starts with a note name
            if any(cleaned_label.startswith(note) for note in ['C','D','E','F','G','A','B']):
                analyzed = self.analyze_chord_with_music21(cleaned_label, key_signature, next_chord)
                if analyzed:
                    return {'symbol': analyzed, 'tags': []}
            
            # Check simple mappings
            if cleaned_label in self.simple_mappings:
                return {'symbol': self.simple_mappings[cleaned_label], 'tags': []}
            
            # Check advanced patterns
            cleaned_lower = cleaned_label.lower()
            for pattern, replacement in self.advanced_patterns.items():
                if re.search(pattern, cleaned_lower):
                    return {'symbol': replacement, 'tags': []}
            
            logger.warning(f"Final harmony analysis failure: '{chord_label}' (Processed: '{cleaned_label}')")
            return {'symbol': f'X:{cleaned_label}', 'tags': ['unanalyzed']}
        
        except Exception as e:
            logger.error(f"Exception during harmony analysis: '{chord_label}' - {e}")
            return {'symbol': f'X:{chord_label}', 'tags': ['unanalyzed']}
    
    def analyze_chord_with_music21(self, chord_label, key_signature, next_chord=None):
        """Advanced harmony analysis using music21 (v4.0 simplified)"""
        key = self.create_key_safely(key_signature)
        chord_obj = self.parse_chord_label_to_music21(chord_label)
        
        if not chord_obj:
            return None

        # Roman numeral analysis is most important
        try:
            roman = m21.roman.romanNumeralFromChord(chord_obj, key)
            return roman.figure
        except Exception:
            logger.debug(ff"Roman numeral analysis failed: {chord_label}")

        # If failed, attempt secondary dominant detection
        secondary_result = self.detect_secondary_dominant_directly(chord_label, key_signature, next_chord)
        if secondary_result:
            return secondary_result
            
        return None
    
    def detect_secondary_dominant_directly(self, chord_label, key_signature, next_chord):
        """Directly detect secondary dominants"""
        try:
            key = self.create_key_safely(key_signature)

            # Extract chord root
            root_match = re.match(r'^([A-G][#b]?)', chord_label)
            if not root_match:
                return None
            
            chord_root = root_match.group(1)
            
            # dominant 타입인지 확인
            is_dominant_type = any(term in chord_label.lower() for term in 
                                 ['dominant', '7', 'seventh']) and 'diminished' not in chord_label.lower()
            
            if not is_dominant_type:
                return None
            
            # Check if it is a dominant type
            secondary_mappings = {
                'I': key.tonic.name,
                'ii': key.pitches[1].name,
                'iii': key.pitches[2].name,
                'IV': key.pitches[3].name,
                'V': key.pitches[4].name,
                'vi': key.pitches[5].name,
            }
            
            for roman, target_root in secondary_mappings.items():
                target_pitch = m21.pitch.Pitch(target_root)
                dominant_of_target = target_pitch.transpose(7).name  # 완전5도 위
                
                if chord_root == dominant_of_target:
                    return f"V7/{roman}" if is_dominant_type else f"V/{roman}"
            
        except Exception as e:
            logger.debug(f"Secondary dominant detection failed: {e}")
        
        return None
    
    def parse_chord_label_to_music21(self, chord_label):
        """Convert normalized text to music21 Chord object (v4.0 simplified)"""
        try:
            # music21's built-in parser handles most complex strings like
            # "C# minor diminished tetrachord" well.
            chord = m21.chord.Chord(chord_label)
            return chord
        except Exception:
            # If failed, extract root only and create basic triad
            root_match = re.match(r'^([A-G][#b]?)', chord_label)
            if root_match:
                root = root_match.group(1)
                logger.debug(f"'{chord_label}' 파싱 실패, 기본 3화음으로 대체: {root}")
                return m21.chord.Chord(root)
        return None
    
    def simple_chord_analysis(self, chord_label, key_signature):
        """Simple chord analysis (fallback)"""
        try:
            key = self.create_key_safely(key_signature)
            
            root_match = re.match(r'^([A-G][#b]?)', chord_label)
            if not root_match:
                return None
            
            root = root_match.group(1)
            root_pitch = m21.pitch.Pitch(root)
            
            # Calculate scale degree in key
            for i, pitch in enumerate(key.pitches):
                if pitch.name == root_pitch.name:
                    scale_degree = i + 1
                    roman_numerals = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII']
                    roman = roman_numerals[scale_degree - 1]
                    
                    # Chord quality
                    if 'minor' in chord_label.lower():
                        roman = roman.lower()
                    elif 'diminished' in chord_label.lower():
                        roman = roman.lower() + '°'
                    
                    # 7th chords
                    if '7' in chord_label:
                        roman = roman + '7'
                    
                    return roman
                    
        except:
            pass
        
        return None

# ==================== MusicXML Parser v5.1 ====================

class MusicXMLToTickParser:
    """Directly convert MusicXML to tick-based representation (v5.1 - All issues fixed)"""
    
    def __init__(self, ticks_per_quarter=480):
        # Unified by TPQ (Ticks Per Quarter)
        self.ticks_per_quarter = ticks_per_quarter
        self.harmony_normalizer = HarmonyNormalizer()
        
    def parse_file(self, filepath: Path) -> Dict:
        """Parse a single MusicXML file and convert it to tick data"""
        logger.info(f"Parsing {filepath}")
        
        try:
            score = m21.converter.parse(str(filepath))
            
            # Separate hands (v3.2: Clef-based)
            try:
                right_hand, left_hand = self.separate_hands_by_clef(score)
            except ValueError as e:
                logger.warning(f"Failed to separate hands {filepath}: {e}")
                return None
            
            # Extract metadata
            metadata = self.extract_metadata(score)
            
            # v5.1: Extract all time signature changes
            time_signatures = self.get_all_time_signatures(score)
            
            # Convert to tick events (v5.1: pass time signature changes list)
            tick_events = self.create_tick_events(score, right_hand, left_hand, time_signatures)
            
            # Organize into measures (v5.1: consider time signature changes)
            measures = self.organize_into_measures(tick_events, time_signatures)
            
            return {
                'metadata': metadata,
                'measures': measures,
                'total_ticks': max(e['tick'] for m in measures for e in m['events']) if measures and measures[0]['events'] else 0
            }
            
        except Exception as e:
            logger.error(f"Error parsing {filepath}: {e}")
            return None
    
    def separate_hands_by_clef(self, score) -> Tuple[m21.stream.Part, m21.stream.Part]:
        """
        v3.2+: Separate hands based on Clef
        TrebleClef = Right Hand
        BassClef = Left Hand
        This is the most musically accurate and robust method.
        """
        right_hand_part = None
        left_hand_part = None
        
        # Check all parts
        for i, part in enumerate(score.parts.stream()):
            # Check the first clef of each part
            # CRITICAL: use flatten() to search through all elements
            first_clef = part.flatten().getElementsByClass(m21.clef.Clef).first()
            
            if first_clef is None:
                logger.warning(f"Could not find clef in part {i+1}")
                continue
            
            # Classify by clef type
            if isinstance(first_clef, m21.clef.TrebleClef):
                if right_hand_part is None:
                    right_hand_part = part
                    logger.debug(f"Part {i+1}: TrebleClef (Right Hand) found")
                else:
                    logger.warning(f"Part {i+1}: Duplicate TrebleClef found, ignoring")
                    
            elif isinstance(first_clef, m21.clef.BassClef):
                if left_hand_part is None:
                    left_hand_part = part
                    logger.debug(f"Part {i+1}: BassClef (Left Hand) found")
                else:
                    logger.warning(f"Part {i+1}: Duplicate BassClef found, ignoring")
                    
            else:
                # Other clefs (AltoClef, TenorClef, etc.)
                logger.info(f"파트 {i+1}: {type(first_clef).__name__} 발견, 피아노 파트가 아님")
        
        ## Check if both parts were successfully identified
        if right_hand_part is not None and left_hand_part is not None:
            logger.info("Successfully identified hands: TrebleClef (RH) + BassClef (LH)")
            return right_hand_part, left_hand_part
        else:
            # If either is missing, it's not a standard piano score
            missing_parts = []
            if right_hand_part is None:
                missing_parts.append("TrebleClef (Right Hand)")
            if left_hand_part is None:
                missing_parts.append("BassClef (Left Hand)")
                
            raise ValueError(
                f"Not a standard piano score. "
                f"Missing parts: {', '.join(missing_parts)}. "
                f"Total parts: {len(score.parts)}"
            )
    
    def get_all_time_signatures(self, score) -> List[Dict]:
        """v5.1: Extract all time signature change information"""
        ts_changes = []
        
        # Find all meter changes
        for ts in score.flatten().getElementsByClass(m21.meter.TimeSignature):
            ts_changes.append({
                'offset': ts.offset,  # quarterLength 단위
                'offset_tick': int(ts.offset * self.ticks_per_quarter),
                'time_sig': ts,
                'numerator': ts.numerator,
                'denominator': ts.denominator
            })
        
        # Sort chronologically
        ts_changes.sort(key=lambda x: x['offset'])
        
        # Default value if no time signature info
        if not ts_changes:
            default_ts = m21.meter.TimeSignature('4/4')
            ts_changes.append({
                'offset': 0,
                'offset_tick': 0,
                'time_sig': default_ts,
                'numerator': 4,
                'denominator': 4
            })
        
        logger.info(f"Found {len(ts_changes)} time signature(s)")
        for i, ts in enumerate(ts_changes):
            logger.info(f"  TS {i+1}: {ts['numerator']}/{ts['denominator']} at offset {ts['offset']}")
        
        return ts_changes
    
    def create_tick_events(self, score, right_hand, left_hand, time_signatures) -> List[TickEvent]:
        """Convert notes to tick events (v5.1: consider time signature changes)"""
        # Temporarily store events for all ticks
        tick_raw_events = defaultdict(lambda: {
            'right': {'note_on': []},
            'left': {'note_on': []},
            'chord_symbol': None,
            'is_phrase_boundary': False,
            'cadence_type': None
        })
        
        # Harmony analysis (v5.1: includes gap filling)
        key = score.analyze('key')
        harmony_data = self.analyze_harmony_detailed_v51(score, key)
        
        # v5.1: Process right hand - support individual velocity
        self.process_hand_notes_v51(right_hand, 'right', tick_raw_events)
        
        # v5.1: Process left hand - support individual velocity
        self.process_hand_notes_v51(left_hand, 'left', tick_raw_events)

        # Add harmony info
        for tick, chord_info in harmony_data.items():
            if tick in tick_raw_events:
                tick_raw_events[tick]['chord_symbol'] = chord_info['symbol']
            
        # Create final tick events (v5.1: detect unclosed notes)
        return self.finalize_tick_events_v51(tick_raw_events)
    
    def analyze_harmony_detailed_v51(self, score, key) -> Dict:
        """v5.1: Harmony analysis - Includes gap filling"""
        harmony_data = {}
        
        # Chordify score - Analyze all notes vertically
        chordified = score.chordify()
        
        # CRITICAL: flatten() to obtain absolute offset
        chord_elements = chordified.flatten().getElementsByClass(m21.chord.Chord)
        
        logger.debug(f"총 {len(chord_elements)}개의 화음 발견")
        
        # List to hold analysis results
        analysis_results = []
        
        # Track previous chord (for context analysis)
        prev_chord_name = None
        last_valid_harmony = {'symbol': 'N.C.', 'tags': []}  # v5.1: 갭 채우기용
        
        for i, chord_element in enumerate(chord_elements):

            start_tick = int(chord_element.offset * self.ticks_per_quarter)
            duration_ticks = int(chord_element.duration.quarterLength * self.ticks_per_quarter)
            end_tick = start_tick + duration_ticks
            

            chord_name = chord_element.pitchedCommonName
            

            next_chord_name = None
            if i + 1 < len(chord_elements):
                next_chord_name = chord_elements[i + 1].pitchedCommonName
            

            normalized = self.harmony_normalizer.normalize_with_context(
                chord_name, 
                str(key), 
                prev_chord_name, 
                next_chord_name
            )
            
            # Fallback Analysis - Switch to direct object analysis if name analysis fails
            if normalized['symbol'].startswith('X:') or normalized['symbol'] == 'X':
                logger.debug(f"'{chord_name}' name analysis failed, switching to direct object analysis")
                
                try:
                    # Check if it is a scale pattern first (these are not chords)
                    scale_patterns = ['tetrachord', 'trichord', 'hexachord', 'pentachord', 'trimirror']
                    if any(pattern in chord_name.lower() for pattern in scale_patterns):
                        normalized = {'symbol': 'X_Scale', 'tags': ['scale_pattern']}
                        logger.debug(f"Classified as scale pattern: {chord_name}")
                    else:
                        # Apply music21's Roman numeral analyzer directly to the chord object
                        roman = m21.roman.romanNumeralFromChord(chord_element, key)
                        normalized['symbol'] = roman.figure
                        normalized['tags'] = ['fallback_analyzed']
                        logger.debug(f"Direct object analysis successful: {chord_name} → {roman.figure}")
                        
                except Exception as fallback_e:
                    # If still fails, record constituent notes of the chord
                    try:
                        pitch_classes = sorted([p.pitchClass for p in chord_element.pitches])
                        pitch_names = [p.name for p in chord_element.pitches]# Check if it is a scale pattern first (these are not chords)
                        normalized['symbol'] = f"PC:{pitch_classes}"
                        normalized['tags'] = ['pitch_class_set', f'notes:{",".join(pitch_names)}']
                        logger.warning(f"Roman numeral analysis failed, recording interval set: {chord_name} → {pitch_classes}")
                    except:
                        # Last resort: preserve original name
                        normalized['symbol'] = f"Unknown:{chord_name}"
                        normalized['tags'] = ['unanalyzed']
                        logger.error(f"All analysis failed: '{chord_name}'")
            
            # Add results to list
            analysis_results.append(normalized)
            
            # Save if valid harmony
            if not normalized['symbol'].startswith(('X', 'PC:', 'Unknown:')):
                last_valid_harmony = normalized
            
            # Assign harmony info to all ticks where this chord persists
            for tick in range(start_tick, end_tick):
                harmony_data[tick] = normalized
            
            # Save current chord as previous chord for next iteration
            prev_chord_name = chord_name
        
        # v5.1: Fill harmony gaps - with previous valid harmony
        max_tick = max(harmony_data.keys()) if harmony_data else 0
        current_harmony = last_valid_harmony
        
        for tick in range(max_tick + 1):
            if tick in harmony_data:
                current_harmony = harmony_data[tick]
            else:
                # If gap found, fill with previous harmony
                harmony_data[tick] = current_harmony
        
        # Output final stats
        total_chords = len(analysis_results)
        if total_chords > 0:
            successful = sum(1 for h in analysis_results 
                           if not h['symbol'].startswith(('X', 'PC:', 'Unknown:')))
            success_rate = (successful / total_chords) * 100
            logger.info(f"Harmony analysis complete: {successful}/{total_chords} ({success_rate:.1f}% success)")
        
        return harmony_data
    
    def process_hand_notes_v51(self, hand_stream, hand_name, tick_raw_events):
        """v5.1: Process notes for one hand - Support individual velocity"""
        # CRITICAL: flatten() to obtain absolute offset
        for element in hand_stream.flatten().notes:
            # Check offset/duration None
            if (element.offset is None or 
                element.duration is None or 
                element.duration.quarterLength is None):
                logger.warning(f"Note/Chord location or duration info missing: {element}, skipping.")
                continue

            # Safe to calculate
            start_tick = int(element.offset * self.ticks_per_quarter)
            duration_ticks = int(element.duration.quarterLength * self.ticks_per_quarter)
            end_tick = start_tick + duration_ticks
            
            if isinstance(element, m21.note.Note):
                pitch = element.pitch.midi
                
                # Handle velocity
                velocity = 80  # 기본값
                if hasattr(element, 'volume') and element.volume:
                    if hasattr(element.volume, 'velocity') and element.volume.velocity is not None:
                        velocity = int(element.volume.velocity)
                
                # NOTE_ON event includes duration info
                note_data = {
                    'pitch': pitch,
                    'velocity': velocity,
                    'start_tick_abs': start_tick,
                    'end_tick_abs': end_tick
                }
                tick_raw_events[start_tick][hand_name]['note_on'].append(note_data)
                
            elif isinstance(element, m21.chord.Chord):
                # v5.1: Check individual velocity for each note in chord
                default_velocity = 80
                if hasattr(element, 'volume') and element.volume:
                    if hasattr(element.volume, 'velocity') and element.volume.velocity is not None:
                        default_velocity = int(element.volume.velocity)
                
                for i, pitch in enumerate(element.pitches):
                    midi_pitch = pitch.midi
                    
                    # v5.1: Check individual note velocity
                    velocity = default_velocity
                    # Include duration info for each note in chord
                    if hasattr(element, 'notes') and i < len(element.notes):
                        note = element.notes[i]
                        if hasattr(note, 'volume') and note.volume:
                            if hasattr(note.volume, 'velocity') and note.volume.velocity is not None:
                                velocity = int(note.volume.velocity)
                    

                    note_data = {
                        'pitch': midi_pitch,
                        'velocity': velocity,
                        'start_tick_abs': start_tick,
                        'end_tick_abs': end_tick
                    }
                    tick_raw_events[start_tick][hand_name]['note_on'].append(note_data)
    
    def finalize_tick_events_v51(self, tick_raw_events) -> List[TickEvent]:
        """v5.1: Convert raw events to final TickEvent - Detect unclosed notes"""
        tick_events = []
        
        max_tick = max(tick_raw_events.keys()) if tick_raw_events else 0
        
        # v5.1: Tracking for unclosed note detection
        unclosed_notes = []
        
        for tick in range(max_tick + 1):
            event = TickEvent(
                tick=tick,
                rh_action='HOLD',
                rh_notes=[],
                lh_action='HOLD', 
                lh_notes=[]
            )
            
            if tick in tick_raw_events:
                raw = tick_raw_events[tick]
                
                # Structural info
                event.chord_symbol = raw['chord_symbol']
                event.is_phrase_boundary = raw['is_phrase_boundary']
                event.cadence_type = raw['cadence_type']
                
                # Process each hand - Only process NOTE_ON
                for hand, prefix in [('right', 'rh'), ('left', 'lh')]:
                    note_on_list = raw[hand]['note_on']
                    
                    if note_on_list:
                        setattr(event, f'{prefix}_action', 'NOTE_ON')
                        # Note object creation includes duration info
                        notes = []
                        for n in note_on_list:
                            note = Note(
                                pitch=n['pitch'], 
                                velocity=n['velocity'],
                                start_tick_abs=n['start_tick_abs'],
                                end_tick_abs=n['end_tick_abs']
                            )
                            notes.append(note)
                            
                            # v5.1: Detect unclosed notes
                            if n['end_tick_abs'] > max_tick + 1000: # Exceeds by 1000+ ticks
                                unclosed_notes.append({
                                    'hand': hand,
                                    'pitch': n['pitch'],
                                    'start': n['start_tick_abs'],
                                    'end': n['end_tick_abs'],
                                    'excess': n['end_tick_abs'] - max_tick
                                })
                        
                        setattr(event, f'{prefix}_notes', notes)
                            
            tick_events.append(event)
        
        # v5.1: Warn about unclosed notes
        if unclosed_notes:
            logger.warning(f"Found {len(unclosed_notes)} unclosed notes!")
            for note in unclosed_notes[:5]:  # Show first 5 only
                logger.warning(
                    f"  {note['hand']} hand pitch {note['pitch']}: "
                    f"starts at {note['start']}, ends at {note['end']} "
                    f"(exceeds by {note['excess']} ticks)"
                )
            if len(unclosed_notes) > 5:
                logger.warning(f"  ... and {len(unclosed_notes) - 5} more")
            
        return tick_events
    
    def organize_into_measures(self, tick_events, time_signatures) -> List[Dict]:
        """v5.1: Organize tick events into measures - Consider time signature changes"""
        measures = []
        
        if not tick_events:
            return measures
        
        max_tick = max(e.tick for e in tick_events)
        current_tick = 0
        bar_num = 1
        ts_index = 0
        
        while current_tick <= max_tick:
            # Find currently applicable time signature
            while (ts_index + 1 < len(time_signatures) and 
                   time_signatures[ts_index + 1]['offset_tick'] <= current_tick):
                ts_index += 1
            
            current_ts = time_signatures[ts_index]
            
            # Calculate measure length based on current time signature
            beats_per_measure = current_ts['numerator']
            beat_duration = 4 / current_ts['denominator']
            ticks_per_measure = int(beats_per_measure * beat_duration * self.ticks_per_quarter)
            
            # Check next time signature change location
            next_ts_tick = float('inf')
            if ts_index + 1 < len(time_signatures):
                next_ts_tick = time_signatures[ts_index + 1]['offset_tick']
            
            # Actual length of this measure (considering time signature changes)
            bar_start = current_tick
            bar_end = min(current_tick + ticks_per_measure, next_ts_tick)
            
            measure_data = {
                'bar_number': bar_num,
                'chord_symbol': None,
                'events': [],
                'time_signature': f"{current_ts['numerator']}/{current_ts['denominator']}"
            }
            
            # Collect events for this measure
            for event in tick_events:
                if bar_start <= event.tick < bar_end:
                    # Relative tick position
                    relative_tick = event.tick - bar_start
                    
                    # Event dictionary including Duration info
                    event_dict = {
                        'tick': event.tick,
                        'relative_tick': relative_tick,
                        'rh_action': event.rh_action,
                        'rh_notes': [{
                            'pitch': n.pitch, 
                            'velocity': n.velocity,
                            'start_tick_abs': n.start_tick_abs,
                            'end_tick_abs': n.end_tick_abs
                        } for n in event.rh_notes],
                        'lh_action': event.lh_action,
                        'lh_notes': [{
                            'pitch': n.pitch, 
                            'velocity': n.velocity,
                            'start_tick_abs': n.start_tick_abs,
                            'end_tick_abs': n.end_tick_abs
                        } for n in event.lh_notes],
                        'chord_symbol': event.chord_symbol,
                        'is_phrase_boundary': event.is_phrase_boundary,
                        'cadence_type': event.cadence_type
                    }
                    measure_data['events'].append(event_dict)
                    
                    # Representative chord of the measure (first chord)
                    if event.chord_symbol and not measure_data['chord_symbol']:
                        measure_data['chord_symbol'] = event.chord_symbol
                        
            if measure_data['events']:
                measures.append(measure_data)
            
            # Move to next measure
            current_tick = bar_end
            bar_num += 1
                
        return measures
    
    def extract_metadata(self, score) -> Dict:
        """Extract metadata (v3.6: Handle None tempo)"""
        metadata = {
            'title': score.metadata.title or 'Unknown',
            'composer': 'Ludwig van Beethoven',
            'movement_title': score.metadata.movementName or '',
        }
        
        # Key
        key = score.analyze('key')
        if key:
            # v3.4: Normalize flat symbol
            key_str = str(key).replace('-', 'b')
            metadata['key_signature'] = key_str
        

        ts = score.flatten().getElementsByClass(m21.meter.TimeSignature)
        if ts:
            metadata['time_signature'] = ts[0].ratioString
        

        tempo = score.flatten().getElementsByClass(m21.tempo.MetronomeMark)
        if tempo:

            if tempo[0].number is not None:
                metadata['tempo'] = int(tempo[0].number)
            else:

                tempo_text = str(tempo[0])
                match = re.search(r'Quarter=(\d+(?:\.\d+)?)', tempo_text)
                if match:
                    metadata['tempo'] = int(float(match.group(1)))
                else:
                    metadata['tempo'] = 120  # 기본값
        else:
            metadata['tempo'] = 120  # 기본값
        

        try:
            metadata['total_measures'] = len(score.parts[0].getElementsByClass(m21.stream.Measure))
        except:
            metadata['total_measures'] = 0
        
        return metadata

# ==================== Batch Processing ====================

def process_beethoven_sonatas(input_dir: Path, output_dir: Path):
    """Process all Beethoven Sonatas v5.1"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    parser = MusicXMLToTickParser()
    
    # Find MusicXML files
    xml_files = list(input_dir.glob("*.xml")) + list(input_dir.glob("*.musicxml")) + list(input_dir.glob("*.mxl"))
    
    logger.info(f"Found {len(xml_files)} MusicXML files")
    
    # Statistics
    success_count = 0
    error_files = []
    non_piano_files = []
    
    # Add stats (v5.1)
    total_note_on_count = 0
    unclosed_notes_files = []
    time_sig_changes_files = []
    
    # Process each file
    for xml_file in tqdm(xml_files, desc="Processing files"):
        try:
            # Parse
            result = parser.parse_file(xml_file)
            
            if result:
                # v5.1: Count NOTE_ON and stats
                file_note_on_count = 0
                file_has_unclosed = False
                
                for measure in result['measures']:
                    for event in measure['events']:
                        if event['rh_action'] == 'NOTE_ON':
                            file_note_on_count += len(event['rh_notes'])
                            # Check for unclosed notes
                            for note in event['rh_notes']:
                                if note['end_tick_abs'] > result['total_ticks'] + 1000:
                                    file_has_unclosed = True
                                    
                        if event['lh_action'] == 'NOTE_ON':
                            file_note_on_count += len(event['lh_notes'])
                            # Check for unclosed notes
                            for note in event['lh_notes']:
                                if note['end_tick_abs'] > result['total_ticks'] + 1000:
                                    file_has_unclosed = True
                
                if file_has_unclosed:
                    unclosed_notes_files.append(xml_file.name)
                
                # Check for time signature changes
                time_sigs = set()
                for measure in result['measures']:
                    if 'time_signature' in measure:
                        time_sigs.add(measure['time_signature'])
                
                if len(time_sigs) > 1:
                    time_sig_changes_files.append((xml_file.name, sorted(time_sigs)))
                
                total_note_on_count += file_note_on_count
                logger.info(f"{xml_file.name}: {file_note_on_count} NOTE_ON events")
                
                # Save
                output_file = output_dir / f"{xml_file.stem}_ticks.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                success_count += 1
            else:
                # None return means hand separation failure (not a standard piano score)
                non_piano_files.append(xml_file.name)
                
        except Exception as e:
            logger.error(f"Failed to process {xml_file}: {e}")
            error_files.append((xml_file.name, str(e)))
    
    # Result Summary
    print(f"\nProcessing Complete!")
    printf"Success: {success_count}/{len(xml_files)}")
    print(f"Non-standard piano scores (excluded): {len(non_piano_files)}")
    if non_piano_files:
        print(f"  {non_piano_files[:5]}..." if len(non_piano_files) > 5 else f"  {non_piano_files}")
    print(f"Errors: {len(error_files)}")
    if error_files:
        for name, error in error_files[:5]:
            print(f"  {name}: {error}")
    
    print(f"\nv5.1 Statistics:")
    print(f"  Total NOTE_ON events: {total_note_on_count:,}")
    print(f"  NOTE_OFF events: 0 (Not generated in v5.1)")
    print(f"  Files with unclosed notes: {len(unclosed_notes_files)}")
    if unclosed_notes_files:
        print(f"    {unclosed_notes_files[:3]}...")
    print(f"  Files with time signature changes: {len(time_sig_changes_files)}")
    if time_sig_changes_files:
        for fname, sigs in time_sig_changes_files[:3]:
            print(f"    {fname}: {' → '.join(sigs)}")
    
    # Save stats file
    stats = {
        'total_files': len(xml_files),
        'success': success_count,
        'non_piano_excluded': non_piano_files,
        'errors': error_files,
        'total_note_on_events': total_note_on_count,
        'unclosed_notes_files': unclosed_notes_files,
        'time_signature_changes': time_sig_changes_files,
        'parser_config': {
            'version': '5.1',
            'major_changes': [
                'Time signature changes handled',
                'Unclosed notes detection',
                'Per-note velocity in chords',
                'Harmony gaps filled'
            ],
            'ticks_per_quarter': 480,
            'harmony_normalizer': 'v4.0.1',
            'hand_separation': 'clef-based (v3.2+)',
            'note_duration': 'included in NOTE_ON events (v5.0+)'
        }
    }
    
    with open(output_dir / 'processing_stats_v51.json', 'w') as f:
        json.dump(stats, f, indent=2)

# ==================== Main Execution ====================

if __name__ == "__main__":
    # Path Configuration
    input_dir = Path("Beethoven_Sonatas_Sliced")
    output_dir = Path("Sliced_Themes_JSON") # New folder name for generation

    print(f"Start MusicXML → Tick Conversion (v5.1)")
    print(f"Input Directory: {input_dir}")
    print(f"Output Directory: {output_dir}")
    print("-" * 60)

    # Execute Processing
    process_beethoven_sonatas(input_dir, output_dir)