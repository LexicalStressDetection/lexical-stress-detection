import torch
import torch.nn as nn


class ResBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.network = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                      padding=padding, stride=stride),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
                      padding=padding, stride=stride),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.network(x)
        out = out + x
        out = self.relu(out)
        return out


class CNNStressNet(nn.Module):

    def __init__(self, reduction='mean'):
        super().__init__()
        self.loss_layer = nn.CrossEntropyLoss(reduction=reduction)
        self.cnn_network = nn.Sequential(
            nn.BatchNorm2d(num_features=3),
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=(3 - 1)//2, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=(3 - 1)//2, stride=2),
            ResBlock(in_channels=32, out_channels=32, kernel_size=3),
            nn.MaxPool2d(kernel_size=3, padding=(3 - 1) // 2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=(0, (3 - 1) // 2), stride=2),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, 4))
        )

        self.dnn_network = nn.Sequential(
            nn.BatchNorm1d(num_features=18),
            nn.Linear(18, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

        self.fully_connected = nn.Sequential(
            nn.BatchNorm1d(num_features=128),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=512),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(num_features=128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, mfcc, non_mfcc):
        n = mfcc.shape[0]
        cnn_out = self.cnn_network(mfcc)
        cnn_out = cnn_out.reshape(n, 64)

        dnn_out = self.dnn_network(non_mfcc)

        out = torch.cat([cnn_out, dnn_out], dim=1)
        out = self.fully_connected(out)

        return out

    def loss(self, predictions, labels):
        loss_val = self.loss_layer(predictions, labels)
        return loss_val
