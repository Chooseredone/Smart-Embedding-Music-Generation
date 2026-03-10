# Smart Embedding Music Generation

This repository contains the complete source code, sample data, and checkpoints for my graduation thesis: *Mathematical Foundations of Polyphonic Music Generation via Structural Inductive Bias and Factorized Representations: A Case Study on Beethoven Piano Sonatas*.

## Overview
- **Thesis Focus**: Demonstrates Smart Embedding (structural inductive bias) for polyphonic music generation, using Beethoven's 32 Piano Sonatas. Key results: 9.47% loss reduction, 28.09% tighter generalization bound, and human Turing Test success (56.6%).
- **Core Components**: Data pipeline (MusicXML → Tokens → Dataset), Model (Conditional Music Transformer with RoPE/ALiBi), Training, Generation, and Analysis (SVD, Nuclear Norm).
- **Reproducibility Note**: To ensure full transparency, we provide scripts for end-to-end processing, sample raw data, processed datasets, and trained checkpoints. Full raw MusicXML from public domain sources (MuseScore by ClassicMan: https://musescore.com/user/19710/sets/54311).

## Installation
1. Clone the repo: git clone [https://github.com/Chooseredone/Smart-Embedding-Music-Generation.git](https://github.com/Chooseredone/Smart-Embedding-Music-Generation.git)
   cd Smart-Embedding-Music-Generation
2. Install dependencies: `pip install -r requirements.txt`
   - Tested on Python 3.10+, PyTorch 2.0+.
   - Optional: Flash Attention for faster training (`pip install flash-attn`).

## Data Preparation
- **Full Raw Data**: Download Beethoven's 32 Piano Sonatas in MusicXML format from MuseScore (public domain, by ClassicMan: https://musescore.com/user/19710/sets/54311). Place in `data/raw_musicxml/`.
- **Samples Provided**:
  - Raw (Original MusicXML): `data/samples/raw/` – Full sonata files from MuseScore (e.g., beethoven_sonata16.mxl). Use for testing the full pipeline from scratch.
  - Parsed (Intermediate): `data/samples/parsed/` – Example parsed JSON (e.g., after musicxml_parser.py on a theme slice).
- **Processed Data**: `data/processed/beethoven_final_perfect.pt` (tokenized dataset from EventStreamBuilder.py, 142 sequences, vocab 1,499) provided.
    - Note: The chunked version (beethoven_chunked.pt, 374 chunks) can be regenerated from this file via the chunking step in beethoven_train.py (or similar fixed-length splitting). If needed, run the training pipeline to create it.
- **Theme Definitions**: `data/theme_definitions.json` defines measure ranges for extraction.
- **Pipeline**:
  1. Extract themes: `python preprocessing/ThemeExtractor.py --definitions data/theme_definitions.json --source data/raw_musicxml/ --output data/extracted_themes/`
  2. Parse to ticks: `python preprocessing/musicxml_parser.py --input_dir data/extracted_themes/ --output_dir data/parsed_json/`
  3. Build dataset: `python preprocessing/EventStreamBuilder.py --json_dir data/parsed_json/ --output_dir data/processed/ --output_name beethoven_final_perfect.pt`

## Training
- Train Smart ON: `python training/beethoven_train.py --data_path data/processed/beethoven_final_perfect.pt --model_size large --use_smart_embedding True --checkpoint checkpoints/smart_on_best.pt`
- Train Smart OFF (ablation): Use `--use_smart_embedding False` for comparison.
- Checkpoints provided: Load with `torch.load('checkpoints/smart_on_best.pt')`.

## Generation and Evaluation
- Generate music: `python generation/beethoven_gen.py --checkpoint checkpoints/smart_on_best.pt --output generated.mid --template [optional template.json] --tempo 100`
- Standardize Ground Truth: `python generation/convert_mxl_to_model_midi.py input.mxl output.mid --checkpoint checkpoints/smart_on_best.pt --start-bar 98 --end-bar 116 --tempo 110`
- Turing Test: Use generated MIDI for human eval (see Appendix F).

## Analysis
- Nuclear Norm: `python analysis/calculate_nuclear_norm.py --checkpoint checkpoints/smart_on_best.pt`
- SVD/Efficiency:

## Checkpoints
- `checkpoints/smart_on_best.pt`: Best Smart ON model (val loss 1.013).
- `checkpoints/smart_off_best.pt`: Smart OFF for comparison (ablation study).

## ⚖️ Pre-trained Models (Reproducibility)
To verify our results (Smart ON vs. Smart OFF) without re-training, we provide the official checkpoints and weights used in the dissertation.
### 📂 [Download All Checkpoints & Weights (Google Drive)](https://drive.google.com/drive/folders/1kNwpzUM15ZmM9oEUpjwRkJ2CIFRpTRQJ?usp=sharing)
The folder contains:
* **Smart ON (Proposed):** `smart_on_best.pt` (Full Checkpoint), `smart_on_weights.pt` (Weights Only)
* **Smart OFF (Baseline):** `smart_off_best.pt` (Full Checkpoint), `smart_off_weights.pt` (Weights Only)
> **Note:** After downloading, please place the `.pt` files in the `checkpoints/` directory to run the generation scripts.

## License
MIT License. See LICENSE file.

Contact: [kennethribet@gmail.com]. For issues, open a GitHub Issue.
