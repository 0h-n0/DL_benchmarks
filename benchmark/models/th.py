import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import tqdm


class Trainer(object):
    def __init__(self, model, ngpu, options=None):
        self.model = model
        self.ngpu = ngpu
        self.gpu_mode = True if ngpu >= 1 else False
        if self.gpu_mode:
            gpus = [i for i in range(self.ngpu)]
            self.model = torch.nn.DataParallel(model, device_ids=gpus)
            self.model.cuda()

        if options['benchmark_mode']:
            torch.backends.cudnn.benchmark = True
            
    def set_optimizer(self, opt_type, opt_conf):
        if opt_type == 'SGD':
            self.optimizer = optim.SGD(self.model.parameters(),
                                       lr=opt_conf['lr'],
                                       momentum=opt_conf['momentum'])
        elif opt_type == 'Adam':
            self.optimizer = optim.Adam(self.model.parameters(),
                                        lr=opt_conf['lr'])
        else:
            raise NotImplementedError
        
    def run(self, iterator, mode='train'):
        report = dict()
        criterion = torch.nn.CrossEntropyLoss().cuda()
        for idx, (x, t) in enumerate(iterator):
            total_s = time.perf_counter()            
            x = torch.FloatTensor(x)
            t = torch.LongTensor(t) 
            if self.gpu_mode:
                x = x.cuda()
                t = t.cuda()
            x, t = Variable(x), Variable(t)
            forward_s = time.perf_counter()
            x = self.model(x)
            forward_e = time.perf_counter()
            self.optimizer.zero_grad()
            loss = criterion(x, t)
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
 
    
