import torch.nn as nn
from neuralop.models import FNO


class FNO_CNN(FNO):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lifting = nn.Sequential(
            nn.Conv2d(self.in_channels+2, self.hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),

            nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),

            nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size=1),
        )

        self.projection = nn.Sequential(
            nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),

            nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),

            nn.Conv2d(self.hidden_channels, self.out_channels, kernel_size=1)
        )
