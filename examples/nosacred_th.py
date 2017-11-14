import sys
import time
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable


class Iterator(object):
    def __init__(self, data_type, image_shape, sequence_shape, niteration,
    batch_size, label_size, target_type):
        self.data_type = data_type
        self.image_shape = image_shape
        self.sequence_shape = sequence_shape
        self.niteration = niteration
        self.batch_size = batch_size
        self.label_size = label_size
        self.target_type = target_type
        self._i = 0
        
    def __iter__(self):
        return self
    
    def __next__(self):
        if self._i == self.niteration:
            raise StopIteration()
        self._i += 1
        
        if self.data_type == 'image':
            ### data dimension = [batch, channel, height, width]
            dims = np.prod(self.image_shape)
            data = np.random.random(dims * self.batch_size)
            data = data.reshape(self.batch_size, *self.image_shape)
            ### target dimension = [batch]
            _target = np.random.randint(self.label_size, size=self.batch_size)
            if self.target_type == 'one-hot':
                target = np.zeros((self.batch_size, self.label_size))
                target[np.arange(self.batch_size), _target] = 1
            else:
                target = _target
                
        elif self.data_type == 'sequence':
            data = np.random.random()
            
        return (data, target)

    def __len__(self):
        return self.niteration

class CNN(nn.Module):
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
 
if __name__ == "__main__":
    ngpu = int(sys.argv[1])
    channel = 3
    xdim = 19
    ydim = 40
    output_num = 3000
    data_type = 'image'
    data_config = dict(
        image_shape = (3, 19, 40), # (channel, witdth, height)
        sequence_shape = 28, # feature
        niteration = 1000,
        batch_size = 2000,
        label_size = 3000,
        target_type = None
    )

    iterator = Iterator(data_type, **data_config)
    torch.backends.cudnn.benchmark = True
    
    gpus = [i for i in range(ngpu)]
    
    model = CNN(channel, xdim, ydim, output_num)
    optimizer = optim.SGD(model.parameters(),
                          lr=0.1,
                          momentum=0.9)

    model = torch.nn.DataParallel(model, device_ids=gpus)
    model.cuda()
    criterion = torch.nn.CrossEntropyLoss().cuda()
    
    model.train()
        
    for idx, (x, t) in enumerate(iterator):
        start = time.time()
        x = torch.FloatTensor(x)
        t = torch.LongTensor(t)
        x, t = Variable(x), Variable(t)        
        x = x.cuda()
        t = t.cuda()

        x = model(x)

        optimizer.zero_grad()
        loss = criterion(x, t)
        loss.backward()
        optimizer.step()
        print(time.time() - start)
