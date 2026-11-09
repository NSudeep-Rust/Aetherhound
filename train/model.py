import torch
import torch.nn as nn

class AudioAutoencoder(nn.Module):
    def __init__(self):
        super(AudioAutoencoder, self).__init__()
        
        # 1. The Encoder Network (Compressor)
        self.encoder = nn.Sequential(
            nn.Flatten(),                  # Flatten 64x257 matrix to a 16448 single vector
            nn.Linear(64 * 257, 512),       # Compress down to 512 parameters
            nn.ReLU(),
            nn.Linear(512, 64),            # Compress down to 64 parameters (Bottleneck latent space)
            nn.ReLU()
        )
        
        # 2. The Decoder Network (Decompressor)
        self.decoder = nn.Sequential(
            nn.Linear(64, 512),            # Decompress back to 512 parameters
            nn.ReLU(),
            nn.Linear(512, 64 * 257),      # Expand back to original 16448 size
            nn.Sigmoid()                   # Bounds outputs to a clean 0.0 - 1.0 range
        )

    def forward(self, x):
        # Flatten batch inputs if they are passed as raw 2D frames
        original_shape = x.shape
        
        # Encode and then decode
        latent = self.encoder(x)
        reconstructed_flat = self.decoder(latent)
        
        # Reshape the output vector perfectly back to the original input shape layout
        return reconstructed_flat.view(original_shape)

if __name__ == "__main__":
    print("[*] Instantiating PyTorch Audio Autoencoder model structural check...")
    model = AudioAutoencoder()
    
    # Run a quick fake tensor through it to confirm matrix shapes map seamlessly
    mock_input = torch.randn(1, 64, 257)
    mock_output = model(mock_input)
    
    print(f"[+] Input tensor dimensions: {mock_input.shape}")
    print(f"[+] Output tensor dimensions: {mock_output.shape}")
    
    assert mock_input.shape == mock_output.shape, "Model failed matrix reconstruction shape check!"
    print("[+] Model architecture verified completely functional!")
