import time
from contextlib import contextmanager

import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

from benchmark.models.base_trainer import BaseTrainer

class Classifier(nn.Module):
    def __init__(self, model, criterion):
        super(Classifier, self).__init__()
        self.model = model
        self.criterion = criterion

    def __call__(self, x, t):
        out = self.model(x)
        loss = self.criterion(out, t)
        return loss
        
        

class Trainer(BaseTrainer):
    def __init__(self, model, ngpu, options,
                 data_options=None, time_options=None):
        self.model = model
        self.ngpu = ngpu
        self.gpu_mode = True if ngpu >= 1 else False
        self.halfmode = options['half']
        self.time_options = time_options
        self._elapsed_time = 0        
        self.criterion = torch.nn.CrossEntropyLoss().cuda()
        self.options = options
        
        if options['benchmark_mode']:
            torch.backends.cudnn.benchmark = True


        if options['parallel_loss']:
            self.model = Classifier(self.model, self.criterion)

        if self.gpu_mode:
            if self.ngpu == 1:
                self.model.cuda()
            else:
                gpus = [i for i in range(self.ngpu)]
                self.model = torch.nn.DataParallel(model, device_ids=gpus)
                self.model.cuda()
                
            if self.halfmode:
                self.model.half()

        if options['mode'] == 'train':
            self.model.train()
        else:
            self.model.eval()
            
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

    def run(self, iterator):
        report = dict()

        time_series = []
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        total_s = time.perf_counter()        
        for idx, (x, t) in enumerate(iterator):
            if self.time_options == 'total':
                start_event.record()
            x = torch.FloatTensor(x)
            t = torch.LongTensor(t) 
            if self.gpu_mode:
                if self.ngpu == 1:
                    x = x.cuda()
                t = t.cuda()
                if self.halfmode:
                    x = x.half()
            x, t = Variable(x), Variable(t)
            self.optimizer.zero_grad()            
            if self.time_options == 'forward':
                with self._record(start_event, end_event):
                    if self.options['parallel_loss']:
                        loss = self.model(x, t)
                    else:
                        x = self.model(x)
                        loss = criterion(x, t)
            else:
                if self.options['parallel_loss']:
                    loss = self.model(x, t)
                else:
                    x = self.model(x)
                    loss = criterion(x, t)
                    
            if self.options['parallel_loss']:
                loss = loss.mean()

            if self.time_options == 'backward':
                with self._record(start_event, end_event):
                    loss.backward()
            else:
                loss.backward()
            self.optimizer.step()
            
            if self.time_options == 'total':
                end_event.record()
                torch.cuda.synchronize()
                self._elapsed_time = start_event.elapsed_time(end_event)/1000
            if isinstance(iterator, tqdm):
                iterator.set_description('{:>10s} :{:10.7f}s/it'.format(self.time_options,
                                                                        self._elapsed_time))            
            time_series.append(self._elapsed_time)
        torch.cuda.synchronize()
        total_e = time.perf_counter()
        report = dict(
            time_series=time_series,
            total=total_e - total_s,
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
 
    
