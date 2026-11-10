import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from train.model import AudioAutoencoder

def train_model(data_path, model_save_path, epochs=15, batch_size=32):
    # 1. Load the binary array patches generated in Phase 4
    raw_data = np.load(data_path)
    
    # Normalize data between 0 and 1 for optimal Sigmoid reconstruction activation
    max_val = raw_data.max() if raw_data.max() > 0 else 1.0
    normalized_data = raw_data / max_val
    
    # Convert numpy matrix tensors to PyTorch floating point Tensors
    tensor_x = torch.tensor(normalized_data, dtype=torch.float32)
    dataset = TensorDataset(tensor_x)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 2. Instantiate our network architecture
    model = AudioAutoencoder()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print(f"[*] Starting Autoencoder optimization workflow ({epochs} Epochs)...")
    model.train()
    
    # 3. Execution training optimization loop
    for epoch in range(epochs):
        running_loss = 0.0
        for batch in dataloader:
            inputs = batch[0]
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs) # Loss measures how closely the reconstruction matches the original
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            
        epoch_loss = running_loss / len(dataset)
        print(f"    Epoch {epoch+1:02d}/{epochs:02d} | Training Loss: {epoch_loss:.6f}")
        
    # 4. Save trained weights configuration parameters to disk
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    print(f"[+] Model optimization complete! Parameters exported to: {model_save_path}")

if __name__ == "__main__":
    train_model(
        data_path="data/baseline_patches.npy",
        model_save_path="models/autoencoder.pt",
        epochs=15,
        batch_size=32
    )
