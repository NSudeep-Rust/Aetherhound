import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from train.model import AudioAutoencoder

def export_to_onnx(model_path, onnx_save_path):
    # 1. Initialize the structure and load our PyTorch weights
    model = AudioAutoencoder()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # 2. Create a dummy tensor that matches our exact patch window size (64 frames x 257 frequencies)
    # This teaches ONNX the input shape path it needs to lock down
    dummy_input = torch.randn(1, 64, 257)
    
    print(f"[*] Exporting target weights matrix from {model_path} to ONNX format...")
    
    # 3. Compile the structural graph to disk (self-contained, no external data sidecar)
    os.makedirs(os.path.dirname(onnx_save_path), exist_ok=True)
    torch.onnx.export(
        model,
        dummy_input,
        onnx_save_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
        external_data=False,
    )

    print(f"[+] ONNX cross-platform binary compiled completely: {onnx_save_path}")

if __name__ == "__main__":
    export_to_onnx("models/autoencoder.pt", "models/autoencoder.onnx")
