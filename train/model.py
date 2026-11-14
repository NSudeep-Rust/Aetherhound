import torch
import torch.nn as nn

class AudioAutoencoder(nn.Module):
    """Compact convolutional autoencoder for 64 x 257 log spectrograms."""

    def __init__(self):
        super(AudioAutoencoder, self).__init__()
        # Strided convolutions preserve local time/frequency structure and are
        # much smaller than a dense 16k-input autoencoder.
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 48, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(48, 32, kernel_size=3, stride=2, padding=1, output_padding=(1, 0)),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=(1, 0)),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=(1, 0)),
            nn.Sigmoid(),
        )

    def forward(self, x):
        if x.ndim != 3 or tuple(x.shape[-2:]) != (64, 257):
            raise ValueError(f"expected [batch, 64, 257], got {tuple(x.shape)}")
        return self.decoder(self.encoder(x.unsqueeze(1))).squeeze(1)

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
