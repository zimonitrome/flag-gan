import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()
        # Fully Connected Layer: 1x100 -> 1x2048
        self.fc1 = nn.Linear(100, 2048)
        self.fc1_bn = nn.BatchNorm2d(2048)

        # Convolutional Layer: 1024x1x2 -> 512x2x4, F=4, S=2, P=1
        self.conv1 = nn.ConvTranspose2d(1024, 512, 4, 2, 1)
        self.conv1_bn = nn.BatchNorm2d(512)

        # Convolutional Layer: 512x2x4 -> 256x4x8, F=4, S=2, P=1
        self.conv2 = nn.ConvTranspose2d(512, 256, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(256)

        # Convolutional Layer: 256x4x8 -> 128x8x16, F=4, S=2, P=1
        self.conv3 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(128)

        # Convolutional Layer: 128x8x16 -> 4x16x32, F=4, S=2, P=1
        self.conv4 = nn.ConvTranspose2d(128, 4, 4, 2, 1)

    def forward(self, x):
        x = F.relu(self.fc1_bn(self.fc1(x)))
        x = x.view(-1, 1024, 1, 2)  # reshape
        x = F.relu(self.conv1_bn(self.conv1(x)))
        x = F.relu(self.conv2_bn(self.conv2(x)))
        x = F.relu(self.conv3_bn(self.conv3(x)))
        x = F.tanh(self.conv4(x))
        return x


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        # Convolutional Layer: 4x16x32 -> 128x8x16, F=4, S=2, P=1
        self.conv1 = nn.Conv2d(4, 128, 4, 2, 1)

        # Convolutional Layer: 128x8x16 -> 256x4x8, F=4, S=2, P=1
        self.conv2 = nn.Conv2d(128, 256, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(256)

        # Convolutional Layer: 256x4x8 -> 512x2x4, F=4, S=2, P=1
        self.conv3 = nn.Conv2d(256, 512, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(512)

        # Convolutional Layer: 512x2x4 -> 1024x1x2, F=4, S=2, P=1
        self.conv4 = nn.Conv2d(512, 1024, 4, 2, 1)
        self.conv4_bn = nn.BatchNorm2d(1024)

        # Convolutional Layer: 1024x1x2 -> 1x2048, F=(1,2), S=1, P=0
        self.conv5 = nn.Conv2d(1024, 1, (1, 2), 1, 0)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)))
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)))
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)))
        x = F.sigmoid(self.conv5(x))
        return x
