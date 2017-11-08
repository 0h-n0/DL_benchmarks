import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import tqdm

class AbstractDNNClass(nn.Module):
    def set_optimizer(self, opt_type, opt_conf):
        if opt_type == 'SGD':
            self.optimizer = optim.SGD(self.parameters(),
                                       lr=opt_conf['lr'],
                                       momentum=opt_conf['momentum'])
        elif opt_type == 'Adam':
            self.optimizer = optim.Adam(self.parameters(),
                                        lr=opt_conf['lr'])
        else:
            raise NotImplementedError

    def train(self, iterator, mode='train'):
        if mode == 'train':
            super(AbstractDNNClass, self).train()
        else:
            self.eval()
            
        report = dict()
        
        for idx, (x, t) in enumerate(iterator):
            total_s = time.perf_counter()            
            x = torch.FloatTensor(x)
            t = torch.LongTensor(t) 
            if self.gpu_mode:
                x = x.cuda()
                t = t.cuda()
            x, t = Variable(x), Variable(t)
            forward_s = time.perf_counter()
            o = self(x)
            forward_e = time.perf_counter()
            
            self.optimizer.zero_grad()
            loss = F.cross_entropy(o, t)
            backward_s = time.perf_counter()
            loss.backward()
            backward_e = time.perf_counter()            
            self.optimizer.step()
            total_e = time.perf_counter()
            report[idx] = dict(
                forward=forward_e - forward_s,
                backward=backward_e - backward_s,
                total=total_e - total_s
            )
        return report
            
class CNN(AbstractDNNClass):
    def __init__(self, channel, xdim, ydim, output_num, gpu_mode):
        self.gpu_mode = gpu_mode
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
            nn.Conv2d(180, 180, (1, 2), stride=1),
            nn.ReLU(),
            nn.Conv2d(180, 180, (1, 1)),
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

