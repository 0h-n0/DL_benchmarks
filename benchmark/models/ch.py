import time
import copy

import torch
import numpy as np
from tqdm import tqdm 
import chainer
from chainer import Chain
import chainer.links as L
import chainer.functions as F
from chainer.function_hooks import TimerHook

from benchmark.models.base_trainer import BaseTrainer


class Trainer(BaseTrainer):
    def __init__(self, model, ngpu, options,
                 data_options=None, time_options=None):
        
        self.ngpu = ngpu
        self.gpu_mode = True if ngpu >= 1 else False
        self.time_options = time_options
        
        if self.gpu_mode:
            self.model = [copy.deepcopy(model).to_gpu(i) for i in range(ngpu)]
        else:
            self.model = model
        if options['benchmark_mode']:            
            chainer.using_config('autotune', True)
            
    def set_optimizer(self, opt_type, opt_conf):
        if opt_type == 'SGD':
            self.optimizer = chainer.optimizers.SGD(lr=opt_conf['lr'])
            self.optimizer.setup(self.model[0])
        elif opt_type == 'MomentumSGD':
            self.optimizer = chainer.optimizers.MomentumSGD(lr=opt_conf['lr'],
                                                            momentum=opt_conf['momentum'])
            self.optimizer.setup(self.model[0])
        elif opt_type == 'Adam':
            self.optimizer = chainer.optimizers.Adam(lr=opt_conf['lr'])
            self.optimizer.setup(self.model[0])
        else:
            raise NotImplementedError
        self.optimizer.use_cleargrads()
        
    def run(self, iterator, mode='train'):
        report = dict()
        time_series = []
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        total_s = time.perf_counter()
        
        for idx, (x, t) in enumerate(iterator):
            if self.time_options == 'total':
                start_event.record()
            if self.gpu_mode:
                x = x.astype(np.float32)
                t = t.astype(np.int32)
                minibatch = len(x) // self.ngpu
                x = [chainer.Variable(
                    chainer.cuda.to_gpu(x[j*minibatch:(j+1)*minibatch], j))
                         for j in range(self.ngpu)]
                t = [chainer.Variable(
                    chainer.cuda.to_gpu(t[j*minibatch:(j+1)*minibatch], j))
                         for j in range(self.ngpu)]
            else:
                x = chainer.Variable(x.astype(np.float32))
                t = chainer.Variable(t.astype(np.int64))
            if self.time_options == 'forward':
                with self._record(start_event, end_event):
                    o = [_model(_x) for _model, _x in zip(self.model, x)]
            else:
                o = [_model(_x) for _model, _x in zip(self.model, x)]
                
            loss = [F.softmax_cross_entropy(_o, _t) for _o, _t in zip(o, t)]
            self.optimizer.target.cleargrads()
            [_model.cleargrads() for _model in self.model]
            
            if self.time_options == 'backward':
                with self._record(start_event, end_event):
                    [(_loss / self.ngpu).backward()
                     for _model, _loss in zip(self.model, loss)]
            else:
                [(_loss / self.ngpu).backward()
                 for _model, _loss in zip(self.model, loss)]
                
            [self.model[0].addgrads(_model) for _model in self.model]
            self.optimizer.update()
            [_model.copyparams(self.model[0]) for _model in self.model]
            
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


class Convblock(Chain):
    def __init__(self, in_ch, out_ch, kernel, stride=1, pooling=False):
        self.pooling = pooling
        super(Convblock, self).__init__()
        with self.init_scope():
            self.conv = L.Convolution2D(in_ch, out_ch, kernel, stride=stride)
            
    def __call__(self, x):
        if self.pooling:
            return F.max_pooling_2d(F.relu(self.conv(x)), (1, 2), stride=2)
        else:
            return F.relu(self.conv(x))            


class CNN(Chain):
    def __init__(self, channel, xdim, ydim, output_num):
        super(CNN, self).__init__()
        with self.init_scope():
            self.conv1 = Convblock(channel, 180, (xdim, 3), 1)
            self.conv2 = Convblock(180, 180, (1, 3), stride=1, pooling=True)
            self.conv3 = Convblock(180, 180, (1, 3), stride=1)
            self.conv4 = Convblock(180, 180, (1, 3), stride=1, pooling=True)
            self.conv5 = Convblock(180, 180, (1, 2), stride=1)
            self.conv6 = Convblock(180, 180, (1, 1), stride=1)
            self.l1 = L.Linear(540, 2048)
            self.l2 = L.Linear(2048, 2048)
            self.l3 = L.Linear(2048, output_num)

    def __call__(self, x):
        h = self.conv1(x)
        h = self.conv2(h)
        h = self.conv3(h)
        h = self.conv4(h)        
        h = self.conv5(h)
        h = self.conv6(h)
        h = self.l1(h)
        h = self.l2(h)
        h = self.l3(h)
        return h
    
