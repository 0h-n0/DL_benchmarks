"""
neon trainers and models
"""
import os
import time
from functools import partial

import torch
from tqdm import tqdm
from neon import NervanaObject
from neon.backends import gen_backend
from neon.backends.backend import Tensor, Block
from neon.models import Model
import neon.layers as L
import neon.transforms as TF
from neon.initializers import Uniform, Gaussian, Constant
from neon.optimizers import GradientDescentMomentum
from benchmark.models.base_trainer import BaseTrainer

class Trainer(BaseTrainer):
    def __init__(self, model, ngpu, options,
                 data_options=None, time_options=None):
        self.model = model(0)
        self.model.set_batch_size(data_options['batch_size'])
        
        self.ngpu = ngpu
        self.gpu_mode = True if ngpu >= 1 else False
        self.time_options = time_options
        if self.ngpu >= 1:
            self.be = gen_backend(backend='nervanagpu', batch_size=data_options['batch_size'])
        else:
            self.be = gen_backend(backend='mkl', batch_size=data_options['batch_size'])
            
    def set_optimizer(self, opt_type, opt_conf):
        if opt_type == 'SGD':
            self.optimizer = GradientDescentMomentum(opt_conf['lr'],
                                                     momentum_coef=opt_conf['momentum'])
        else:
            raise NotImplementedError
        
    def run(self, iterator, mode='train'):
        report = dict()

        time_series = []
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        self.loss = L.GeneralizedCost(costfunc=TF.CrossEntropyMulti())
        total_s = time.perf_counter()        
        for idx, (x, t) in enumerate(iterator):
            if self.time_options == 'total':            
                start_event.record()
            x = self.be.array(x)
            t = self.be.array(t)
            self.be.begin(Block.minibatch, idx)

            if self.time_options == 'forward':
                with self._record(start_event, end_event):
                    x = self.model(x)
            else:
                x = self.model(x)

            self.total_cost[:] = self.total_cost + self.loss.get_cost(x, t)

            # deltas back propagate through layers
            # for every layer in reverse except the 0th one
            loss = self.loss.get_errors(x, t)
            
            if self.time_options == 'backward':
                with self._record(start_event, end_event):
                    self.model.backward(loss)                    
            else:
                self.model.backward(loss)

            self.optimizer.optimize(self.model.layers_to_optimize, epoch=0)
                
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


class CNN(NervanaObject):
    def __init__(self, channel, xdim, ydim, output_num):
        self.channel = channel
        self.xdim = xdim
        self.ydim = ydim
        self.output_num = output_num
        self.layers = self.set_layers()

    def __call__(self, x, inference=False):
        out = self.layers.fprop(x, inference)
        self.be.convert_data(out, False)
        return out

    @property
    def layers_to_optimize(self):
        return self.layers.layers_to_optimize

    def backward(self, loss):
        out = self.layers.bprop(loss)
        self.be.convert_data(out, False)
        return out
    
    def loss(self, func, x, t):
        return func(x, t)

    def update(self):
        pass
    
    def set_layers(self):
        init_uni = Uniform(low=-0.1, high=0.1)        
        layers = [L.Conv(fshape=(self.channel, self.xdim, 3),
                         init=init_uni, activation=TF.Rectlin())]
        layers = L.Sequential(layers)
        layers.propagate_parallelism("Data")
        self.layers = layers
