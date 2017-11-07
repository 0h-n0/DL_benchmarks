from chainer import c


class CNN(Chain):
    def __init__(self, output_num):
        super(CNN, self).__init__(
            conv1=L.Convolution2D(3, 180, (19, 3), stride=1),
            conv1a=L.Convolution2D(180, 180, (1, 3)),
            conv2=L.Convolution2D(180, 180, (1, 3), stride=1),
            conv2a=L.Convolution2D(180, 180, (1, 3)),
            conv3=L.Convolution2D(180, 180, (1, 3), stride=1),
            conv3a=L.Convolution2D(180, 180, (1, 3)),
            fc4=L.Linear(540, 2048),
            fc5=L.Linear(2048, 2048),
            fc6=L.Linear(2048, output_num)
        )

    def __call__(self, x):
        h = F.relu(self.conv1(x))
        h = F.max_pooling_2d(F.relu(self.conv1a(h)), (1, 2), stride=2)
        h = F.relu(self.conv2(h))
        h = F.max_pooling_2d(F.relu(self.conv2a(h)), (1, 2), stride=2)
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv3a(h))
        h = F.relu(self.fc4(h))
        h = F.relu(self.fc5(h))
        y = self.fc6(h)
        return y
