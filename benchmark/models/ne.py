"""
neon trainers and models
"""
import os
from functools import partial

import torch
from tqdm import tqdm
from neon.backends import gen_backend
from neon.backends.backend import Tensor
import neon.layers as L
import neon.transforms as TF
from neon.initializers import Gaussian
from benchmark.models.base_trainer import BaseTrainer

class Trainer(BaseTrainer):
    def __init__(self, model, ngpu, options,
                 data_options=None, time_options=None):
        self.model = model
        self.ngpu = ngpu
        if self.ngpu >= 1:
            self.be = gen_backend(backend='gpu')
        else:
            self.be = gen_backend(backend='mkl')            
            
    def set_optimizer(self, opt_type, opt_conf):
        if opt_type == 'SGD':
            raise NotImplementedError            
        else:
            raise NotImplementedError
        
    def run(self, iterator, mode='train'):
        report = dict()

        time_series = []
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
            
        for idx, (x, t) in enumerate(iterator):
            if self.time_options == 'total':            
                start_event.record()
            x = self.be.array(x)
            t = self.be.array(t)            
            if self.time_options == 'forward':
                with self._record(start_event, end_event):
                    #self.module.forward(batch, is_train=True)                    
            else:
                #self.module.forward(batch, is_train=True)                

            self.module.update_metric(self.metric, batch.label)
            if self.time_options == 'backward':
                with self._record(start_event, end_event):
                    self.module.backward()
            else:
                self.module.backward()
                    
            self.module.update()

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


class CNN(object):
    def __init__(self, channel, xdim, ydim, output_num):
        self.cnn = partial(cnn,
                           channel=channel,
                           xdim=xdim,
                           ydim=ydim,
                           output_num=output_num,
                           init_norm=Gaussian(loc=0.0, scale=0.01),
        )
        
    def get_func(self):
        return self.cnn
    
    def __call__(self, x):
        return self.cnn(x)
    

def cnn(x, channel, xdim, ydim, output_num, init_norm):
    layers = [
        L.Conv((), init=Gaussian(scale=0.01, bias=Constant(0),
                                 activation=Rectlin())),
        
    ]
    
    layers.append(L.Affine(nout=100, init=init_norm, activation=Rectlin()))
    layers.append(Affine(nout=10, init=init_norm, activation=Softmax()))
    
    return net
