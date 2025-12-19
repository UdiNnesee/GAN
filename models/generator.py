import torch.nn as nn
from .denoising import DenoisingBlock

class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.denoise = DenoisingBlock(3)

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2,
                               padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, 7, padding=3),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.denoise(x)
        x = self.encoder(x)
        x = self.decoder(x)
        return x
