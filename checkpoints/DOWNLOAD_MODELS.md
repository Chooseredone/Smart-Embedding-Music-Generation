# Pre-trained Model Checkpoints

Due to file size limits on GitHub, the pre-trained model checkpoints are hosted on Google Drive. These files correspond to the results presented in **Table 6.1** of the dissertation.

## 🔗 Quick Link
👉 **[Access Google Drive Folder (All Files)](https://drive.google.com/drive/folders/1kNwpzUM15ZmM9oEUpjwRkJ2CIFRpTRQJ?usp=sharing)**

## 📄 File Descriptions

| Model Configuration | File Name | Type | Description |
| :--- | :--- | :--- | :--- |
| **Smart ON (Proposed)** | `smart_on_best.pt` | Full Checkpoint | Contains model weights, optimizer state, and training history. Use for **resuming training**. |
| | `smart_on_weights.pt` | Weights Only | Lightweight file. Use for **inference/generation**. |
| **Smart OFF (Baseline)** | `smart_off_best.pt` | Full Checkpoint | Baseline model for ablation study comparison. |
| | `smart_off_weights.pt` | Weights Only | Baseline weights for generation comparison. |

## 🛠️ How to Use
1. Download the files from the link above.
2. Place them in this `checkpoints/` directory.
3. Run the generation script:
   ```bash
   # Example: Generate using Smart ON weights
   python generation/beethoven_gen.py --checkpoint checkpoints/smart_on_weights.pt --output output.mid