import torch
import numpy as np
from sklearn.metrics import normalized_mutual_info_score
import re
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load data
data = torch.load('output/beethoven_chunked.pt', weights_only=False)  # Adjust path
sequences = data['sequences']
inv_vocab = data['inv_vocab']
pad_id = data['vocab']['<PAD>']  # Assume <PAD> ID

# Extract pitch/hand from NOTE_ON
pitches = []
hands = []
note_on_pattern = re.compile(r'(RH|LH)_NOTE_ON_(\d+)')  # RH/LH only

print("Extracting from 374 chunks...")
total_notes = 0
for seq_idx, seq in enumerate(sequences):
    for token_id in seq:
        if token_id.item() == pad_id:
            continue
        
        token_str = inv_vocab.get(token_id.item(), '')
        match = note_on_pattern.match(token_str)
        
        if match:
            hand_str = match.group(1)  # 'RH' or 'LH'
            pitch_val = int(match.group(2))  # MIDI pitch
            hands.append(hand_str) # hands.append(hand_str) # Save as 'RH' or 'LH' string
            pitches.append(pitch_val)
            total_notes += 1

print(f"Total NOTE_ON events: {total_notes}")

if len(pitches) > 0 and len(set(pitches)) > 1 and len(set(hands)) > 1:
    nmi = normalized_mutual_info_score(pitches, hands)
    print(f"\nNMI I(pitch; hand): {nmi:.4f}")
    if nmi < 0.1:
        print("Strong evidence of independence: Additive decomp valid.")
    elif nmi < 0.3:
        print("Weak correlation: Functional independence holds for Smart design.")
    else:
        print("Moderate correlation: Consider multiplicative decomp alternative.")

    # --- Start of graph generation ---
    print("\nGenerating joint distribution plot...")
    
    # 1. Convert data to pandas DataFrame
    df = pd.DataFrame({'pitch': pitches, 'hand': hands})
    
    # 2. Calculate Joint distribution (using crosstab)
    joint_dist = pd.crosstab(df['pitch'], df['hand'], normalize='all')
    
    # 3. Visualize with Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(joint_dist, cmap="viridis", cbar_kws={'label': 'Proportion of Events'})
    plt.title('Joint Distribution of Pitch and Hand')
    plt.xlabel('Hand')
    plt.ylabel('Pitch (MIDI Note Number)')

    # 4. Save to file
    plot_filename = 'mi_joint_distribution.png'
    plt.savefig(plot_filename)
    
    print(f"Plot saved to {plot_filename}")
    -	# --- End of graph generation ---

else:
    print("Insufficient data for MI calc—need diverse pitches/hands.")