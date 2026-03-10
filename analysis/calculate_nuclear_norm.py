import torch
import argparse

def calculate_nuclear_norm(checkpoint_path):
    """
    Loads a PyTorch checkpoint, extracts the main token embedding matrix,
    and calculates its Nuclear Norm.
    """
    try:
        # Load the checkpoint file. Using map_location='cpu' ensures it works without a GPU.
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print(f"Successfully loaded checkpoint from: {checkpoint_path}")

        # The model state dictionary is often under the key 'model_state_dict' or 'model'.
        # If the file is just the state dict, we can access weights directly.
        # Based on logs and your code, the key for the embedding matrix is 'token_embedding.weight'.
        embedding_matrix_key = 'embedding.token_embedding.weight'
        
        if embedding_matrix_key in checkpoint:
            embedding_matrix = checkpoint[embedding_matrix_key]
        elif 'model' in checkpoint and embedding_matrix_key in checkpoint['model']:
            embedding_matrix = checkpoint['model'][embedding_matrix_key]
        elif 'model_state_dict' in checkpoint and embedding_matrix_key in checkpoint['model_state_dict']:
             embedding_matrix = checkpoint['model_state_dict'][embedding_matrix_key]
        else:
            print(f"Error: Could not find the embedding matrix with key '{embedding_matrix_key}'.")
            print("Please check the keys in your .pt file.")
            return

        print(f"Found embedding matrix of shape: {embedding_matrix.shape}")

        # Calculate the singular values of the matrix.
        # svdvals returns a 1D tensor of singular values.
        singular_values = torch.linalg.svdvals(embedding_matrix.float())

        # The Nuclear Norm is the sum of the singular values.
        nuclear_norm = torch.sum(singular_values)

        print("-" * 30)
        print(f"Nuclear Norm: {nuclear_norm.item():.4f}")
        print("-" * 30)

    except FileNotFoundError:
        print(f"Error: Checkpoint file not found at {checkpoint_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate the Nuclear Norm of an embedding matrix from a PyTorch checkpoint.")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the model checkpoint .pt file.')
    args = parser.parse_args()
    
    calculate_nuclear_norm(args.checkpoint)