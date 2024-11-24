import cv2
import numpy as np
import torch
import torch.nn as nn

class ColorizationNet(nn.Module):
    def __init__(self):
        super(ColorizationNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 2, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

gray_image = cv2.imread('gray_image.jpg', 0)
gray_image = cv2.resize(gray_image, (256, 256))
gray_tensor = torch.tensor(gray_image, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0

model = ColorizationNet()
output = model(gray_tensor)
