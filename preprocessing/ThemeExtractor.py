import json
import argparse
import logging
from pathlib import Path
from typing import Optional  # Fix: Import typing library for 'Optional' type hint
import music21 as m21
from tqdm import tqdm

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ThemeExtractor:
    """
    Extracts specific measure ranges from MusicXML files based on a JSON definition file.
    """
    def __init__(self, definitions_path: str, source_dir: str, output_dir: str):
        self.definitions_path = Path(definitions_path)
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.themes = []

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_definitions(self):
        """Loads and validates the definition file."""
        logger.info(f"Loading theme definitions from: {self.definitions_path}")
        try:
            with open(self.definitions_path, 'r', encoding='utf-8') as f:
                self.themes = json.load(f)
            logger.info(f"Successfully loaded {len(self.themes)} theme definitions.")
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Failed to load or parse definition file: {e}")
            raise

    def find_source_file(self, source_filename: str) -> Optional[Path]:
        """Finds a file with .mxl or .xml extension in the source directory."""
        base_name = Path(source_filename).stem
        for ext in ['.mxl', '.xml', '.musicxml']:
            path = self.source_dir / f"{base_name}{ext}"
            if path.exists():
                return path
        return None

    def extract_themes(self):
        """Extracts and saves all defined themes."""
        if not self.themes:
            logger.warning("No themes defined. Nothing to extract.")
            return

        logger.info("Starting theme extraction process...")
        for theme_info in tqdm(self.themes, desc="Extracting Themes"):
            try:
                source_filename = theme_info['source_file']
                theme_name = theme_info['theme_name']
                start_measure = theme_info['start_measure']
                end_measure = theme_info['end_measure']

                # v1.1 Fix: Respect the start_measure value provided by the user as is.
                # Correctly handles measure 0 (pickup measure) as well.

                # Find source file
                source_path = self.find_source_file(source_filename)
                if not source_path:
                    logger.warning(f"Source file not found for '{source_filename}', skipping.")
                    continue

                # Parse MusicXML
                score = m21.converter.parse(source_path)

                # Extract measure range
                extracted_measures = score.measures(start_measure, end_measure)

                # Create a new score object
                new_score = m21.stream.Score()
                
                # Attempt to copy metadata
                if score.metadata:
                    new_score.metadata = score.metadata
                    new_score.metadata.title = f"{score.metadata.title} - {theme_name}"

                # Insert measures by part
                for part in extracted_measures.parts:
                    new_score.insert(0, part)

                # Generate output filename
                output_filename = f"{Path(source_filename).stem}_{theme_name}.mxl"
                output_path = self.output_dir / output_filename

                # Save as a new MusicXML file
                new_score.write('musicxml', fp=output_path)
                logger.debug(f"Successfully extracted and saved: {output_path.name}")

            except Exception as e:
                logger.error(f"Failed to process theme '{theme_info.get('theme_name')}': {e}")
        
        logger.info("Theme extraction process completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract musical themes from MusicXML files based on a JSON definition.")
    parser.add_argument('--definitions', type=str, required=True, help='Path to the theme_definitions.json file.')
    parser.add_argument('--source', type=str, required=True, help='Directory containing the full source MusicXML files.')
    parser.add_argument('--output', type=str, required=True, help='Directory to save the extracted theme MusicXML files.')

    args = parser.parse_args()

    extractor = ThemeExtractor(
        definitions_path=args.definitions,
        source_dir=args.source,
        output_dir=args.output
    )
    extractor.load_definitions()
    extractor.extract_themes()