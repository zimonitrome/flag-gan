import torch.nn as nn

class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()
        
        self.conv = nn.Sequential(
            # Convolutional Layer: 100x1x1 -> 512x1x2, F=(1,2), S=1, P=0
            nn.ConvTranspose2d(100, 512, (1, 2), 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            # Convolutional Layer: 512x1x2 -> 256x2x4, F=4, S=2, P=1
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            # Convolutional Layer: 256x2x4 -> 128x4x8, F=4, S=2, P=1
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            # Convolutional Layer: 128x4x8 -> 64x8x16, F=4, S=2, P=1
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # Convolutional Layer: 64x8x16 -> 32x16x32, F=4, S=2, P=1
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            # Convolutional Layer: 32x8x16 -> 16x32x64, F=4, S=2, P=1
            nn.ConvTranspose2d(32, 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            # Convolutional Layer: 16x32x64 -> 8x64x128, F=4, S=2, P=1
            nn.ConvTranspose2d(16, 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),

            # Convolutional Layer: 8x64x128 -> 4x128x256, F=4, S=2, P=1
            nn.ConvTranspose2d(8, 4, 4, 2, 1, bias=False)
            # Removed sigmoid for now
        )
            
    def forward(self, x):
        x = self.conv(x)
        return x


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv = nn.Sequential(
            # Convolutional Layer: 4x128x256 -> 8x64x128, F=4, S=2, P=1
            nn.Conv2d(4, 8, 4, 2, 1, bias=False),
            nn.LeakyReLU(),

            # Convolutional Layer: 8x64x128 -> 16x32x64, F=4, S=2, P=1
            nn.Conv2d(8, 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),

            # Convolutional Layer: 16x32x64 -> 32x16x32, F=4, S=2, P=1
            nn.Conv2d(16, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),

            # Convolutional Layer: 32x16x32 -> 64x8x16, F=4, S=2, P=1
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),

            # Convolutional Layer: 64x8x16 -> 128x4x8, F=4, S=2, P=1
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),

            # Convolutional Layer: 128x4x8 -> 256x2x4, F=4, S=2, P=1
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),

            # Convolutional Layer: 256x2x4 -> 512x1x2, F=4, S=2, P=1
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),

            # Convolutional Layer: 512x1x2 -> 1x1x1, F=(1,2), S=1, P=0
            nn.Conv2d(512, 1, (1, 2), 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv(x)
        return x
