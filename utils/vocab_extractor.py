#!/usr/bin/env python3
"""
Extract vocabulary from model checkpoint file
"""

import torch
import json
import argparse
from pathlib import Path

def extract_vocab_from_checkpoint(checkpoint_path: str, output_path: str = "vocab.json"):
    """
    Extract vocab dictionary from PyTorch checkpoint.
    
    Args:
        checkpoint_path: Path to .pt or .pth file
        output_path: Where to save vocab.json
    """
    print(f"📦 Loading checkpoint: {checkpoint_path}")
    
    try:
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print(f"✅ Checkpoint loaded successfully")
        
        # Check available keys
        print(f"📋 Available keys: {list(checkpoint.keys())}")
        
        # Try to find vocab
        vocab = None
        if 'vocab' in checkpoint:
            vocab = checkpoint['vocab']
        elif 'vocabulary' in checkpoint:
            vocab = checkpoint['vocabulary']
        elif 'tokenizer' in checkpoint and 'vocab' in checkpoint['tokenizer']:
            vocab = checkpoint['tokenizer']['vocab']
        elif 'config' in checkpoint and 'vocab' in checkpoint['config']:
            vocab = checkpoint['config']['vocab']
        else:
            print("❌ No vocab found in standard locations")
            print("   Trying to extract from model config...")
            
            # Sometimes vocab_size is stored but not the actual vocab
            if 'config' in checkpoint:
                config = checkpoint['config']
                print(f"   Config: {config}")
        
        if vocab:
            print(f"✅ Found vocabulary with {len(vocab)} tokens")
            
            # Save to JSON
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(vocab, f, indent=2, ensure_ascii=False)
            
            print(f"💾 Saved to: {output_path}")
            
            # Show sample tokens
            sample_keys = list(vocab.keys())[:10]
            print(f"📝 Sample tokens: {sample_keys}")
        else:
            print("⚠️  Could not find vocab. You may need to generate it from training data.")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Extract vocab from checkpoint")
    parser.add_argument('--checkpoint', required=True, help='Path to .pt file')
    parser.add_argument('--output', default='vocab.json', help='Output path')
    
    args = parser.parse_args()
    extract_vocab_from_checkpoint(args.checkpoint, args.output)

if __name__ == '__main__':
    main()
