import torch
import torch.nn as nn

class ESPCN(nn.Module):

    def __init__(self, scale_factor=4):

        super(ESPCN, self).__init__()

        self.feature_extractor = nn.Sequential(

            nn.Conv2d(3, 64, 5, padding=2),
            nn.ReLU(),

            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),

            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),

            nn.Conv2d(
                32,
                3 * (scale_factor ** 2),
                3,
                padding=1
            )
        )

        self.pixel_shuffle = nn.PixelShuffle(
            scale_factor
        )

    def forward(self, x):

        x = self.feature_extractor(x)

        x = self.pixel_shuffle(x)

        return x