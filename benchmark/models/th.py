import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import tqdm
from chainer.function_hooks import TimerHook

class Trainer(object):
    def __init__(self, model, ngpu, options=None):
        self.model = model
        self.ngpu = ngpu
        self.gpu_mode = True if ngpu >= 1 else False
        self.halfmode = options['half']
        
        if options['benchmark_mode']:
            torch.backends.cudnn.benchmark = True
        
        if self.gpu_mode:
            if self.ngpu == 1:
                self.model.cuda()
            else:
                gpus = [i for i in range(self.ngpu)]
                self.model = torch.nn.DataParallel(model, device_ids=gpus)
                self.model.cuda()
            if self.halfmode:
                self.model.half()
            
            
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
        if mode == 'train':
            self.model.train()
        
        for idx, (x, t) in enumerate(iterator):
            start = time.time()
            x = torch.FloatTensor(x)
            t = torch.LongTensor(t) 
            if self.gpu_mode:
                if self.ngpu == 1:
                    x = x.cuda()
                t = t.cuda()
                if self.halfmode:
                    x = x.half()
                    
            x, t = Variable(x), Variable(t)
            x = self.model(x)

            self.optimizer.zero_grad()
            loss = criterion(x, t)
            loss.backward()
            self.optimizer.step()
            print(time.time() - start)
            if idx == 0:
                torch.cuda.synchronize()
                total_s = time.perf_counter()
                len_t = len(t)
            #total_e = time.perf_counter()
            #report[idx] = dict(
            #    total=total_e - total_s
            #)
            #iterator.set_description("total time {:10.7f}".format(total_e - total_s))
            #print("total time {:10.7f}".format(total_e - total_s))
        torch.cuda.synchronize()
        total_e = time.perf_counter()
        report = dict(
            total=total_e - total_s,
            perit=(total_e - total_s)/(idx),
            timepersample=(total_e - total_s)/(idx*len_t),            
            )
        print(report)
            
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
 
    
