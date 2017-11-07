import torch
import torch.nn as nn


class AbstractDNNClass(nn.Module):
    def set_optimizer(self):
        pass
    
    def train(self, iterator):
        for i in iterator:
            print(i.shape)

class CNN(AbstractDNNClass):
    def __init__(self, channel, xdim, ydim, output_num):
        super(CNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channel, 180, (xdim, 3), stride=1),
            nn.ReLU(),
            nn.Conv2d(180, 180, (1, 3)),
            nn.ReLU(),
            nn.MaxPool2d((1, 2), stride=2, ceil_mode=True),
            nn.Conv2d(180, 180, (1, 3), stride=1),
            nn.ReLU(),
            nn.Conv2d(180, 180, (1, 3)),
            nn.ReLU(),
            nn.MaxPool2d((1, 2), stride=2, ceil_mode=True),
            nn.Conv2d(180, 180, (1, 3), stride=1),
            nn.ReLU(),
            nn.Conv2d(180, 180, (1, 3)),
            nn.ReLU())
        self.fc = nn.Sequential(
            nn.Linear(540, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, output_num))

    def forward(self, x):
        h = self.conv(x)
        h = h.view(len(h), -1)
        return self.fc(h)

