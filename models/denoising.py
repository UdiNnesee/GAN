import torch.nn as nn

class DenoisingBlock(nn.Module):
    def __init__(self, channels=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, channels, 3, padding=1)
        )

    def forward(self, x):
        return x + self.net(x)  # residual
