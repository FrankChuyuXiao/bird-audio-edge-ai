
import torch
import torch.nn as nn


class BirdCNN(nn.Module):

    def __init__(self, num_classes):

        super().__init__()

        self.conv_layers = nn.Sequential(

            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                16,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2)
        )

        self.fc = nn.Sequential(

            nn.AdaptiveAvgPool2d((4,4)),

            nn.Flatten(),

            nn.Linear(
                32*4*4,
                num_classes
            )
        )


    def forward(self, x):

        x = self.conv_layers(x)

        x = self.fc(x)

        return x
