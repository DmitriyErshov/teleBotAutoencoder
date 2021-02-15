# import numpy as np
#
import torch
import torch.nn as nn
# import torch.optim as optim
# import torchvision
import torch.nn.functional as F
#
#
#
# seed = 42
#
#
#свёрточный автоэнкодер с одним скрытым слоем
class AE(nn.Module):
    def __init__(self, input_shape):
        super().__init__()

        # Encoder
        self.conv1 = nn.Conv2d(3, 26, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(26)
        self.pool = nn.MaxPool2d(2, 2)

        # Decoder
        self.t_conv3 = nn.ConvTranspose2d(26, 3, 3, stride=2)

    def encode(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        self.bn1
        return x

    def decode(self, x):
        x = F.sigmoid(self.t_conv3(x))
        return x

    def forward(self, x):
        code = self.encode(x)
        return self.decode(code)
#
#
import torch.nn.functional as F


#свёрточный автоэнкодер с двумя скрытыми слоями
class AE2(nn.Module):
    def __init__(self, input_shape):
        super().__init__()

        # Encoder
        self.conv1 = nn.Conv2d(3, 26, 3, padding=1)
        self.conv2 = nn.Conv2d(26, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(16, 4, 3, padding=1)

        # Decoder
        self.t_conv1 = nn.ConvTranspose2d(4, 16, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(16, 26, 2, stride=2)
        self.t_conv3 = nn.ConvTranspose2d(26, 3, 2, stride=1)

    def encode(self, x):
        x = F.relu(self.conv1(x))
        # print(x.shape)
        x = self.pool(x)
        # print(x.shape)
        x = F.relu(self.conv2(x))
        # print(x.shape)
        x = self.pool(x)
        # print(x.shape)
        x = F.relu(self.conv3(x))
        # print(x.shape)
        return x

    def decode(self, x):
        x = F.relu(self.t_conv1(x))
        x = F.sigmoid(self.t_conv2(x))
        x = F.sigmoid(self.t_conv3(x))
        return x

    def forward(self, x):
        code = self.encode(x)
        return self.decode(code)