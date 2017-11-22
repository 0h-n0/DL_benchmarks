"""
mxnet trainers and models
"""
import os
import time
from functools import partial

import torch
import mxnet as mx
from tqdm import tqdm

from benchmark.models.base_trainer import BaseTrainer


class Trainer(BaseTrainer):
    def __init__(self, model, ngpu, options,
                 data_options=None, time_options=None):
        self.model = model
        self.ngpu = ngpu
        self.gpu_mode = True if ngpu >= 1 else False
        self.time_options = time_options
        if self.gpu_mode:
            self.gpus = [mx.gpu(i) for i in range(ngpu)]
            
        if options['benchmark_mode']:
            os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '1'
            os.environ['MXNET_BACKWARD_DO_MIRROR'] = '1'
            ## Mirror mode reduces memory trasferation cost.
            
        data = mx.sym.var('data')
        self.module = mx.mod.Module(symbol=self.model(data),
                               context=self.gpus,
                               data_names=['data'],
                               label_names=['softmax_label'])

        B = data_options['batch_size']
        C, H, W = data_options['image_shape']
        self.data_shape = (B, C, H, W)
        self.label_shape = (B,)

        self.module.bind(data_shapes=zip(['data'], [self.data_shape]),
                         label_shapes=zip(['softmax_label'], [self.label_shape]))
                               
        self.module.init_params(initializer=mx.init.Xavier(magnitude=2.))
            
    def set_optimizer(self, opt_type, opt_conf):
        if opt_type == 'SGD':
            self.opt_type = 'sgd'
            self.lr = opt_conf['lr']
            self.metric = mx.metric.CrossEntropy()
            self.module.init_optimizer(optimizer=self.opt_type,
                                       optimizer_params=(('learning_rate', self.lr),))
            self.metric.reset()
        else:
            raise NotImplementedError
        
    def run(self, iterator, mode='train'):
        report = dict()

        # setup mxnet module
        # https://mxnet.incubator.apache.org/tutorials/basic/data.html        
        ## end setup
        
        time_series = []
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        total_s = time.perf_counter()            
        for idx, (x, t) in enumerate(iterator):
            if self.time_options == 'total':            
                start_event.record()
            x = [mx.nd.array(x)] #[mx.nd.array(x[i, ...].reshape(1, C, H, W)) for i in range(B)]                
            t = [mx.nd.array(t)] #[mx.nd.array([t[i]]) for i in range(B)]
                
            batch = mx.io.DataBatch(x, t)
            if self.time_options == 'forward':
                with self._record(start_event, end_event):
                    self.module.forward(batch, is_train=True)                    
            else:
                self.module.forward(batch, is_train=True)                
            

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
                           output_num=output_num)
        
    def get_func(self):
        return self.cnn
    
    def __call__(self, x):
        return self.cnn(x)
    

def cnn(x, channel, xdim, ydim, output_num):
    net = mx.sym.Convolution(data=x, kernel=(xdim, 3), num_filter=180)
    net = mx.sym.Activation(data=net, act_type='relu')    
    net = mx.sym.Convolution(data=net, kernel=(1, 3), num_filter=180)
    net = mx.sym.Activation(data=net, act_type='relu')        
    net = mx.sym.Pooling(data=net, pool_type='max', kernel=(1, 2), stride=(2, 2))
    net = mx.sym.Convolution(data=net, kernel=(1, 3), num_filter=180)
    net = mx.sym.Activation(data=net, act_type='relu')        
    net = mx.sym.Convolution(data=net, kernel=(1, 3), num_filter=180)
    net = mx.sym.Activation(data=net, act_type='relu')        
    net = mx.sym.Pooling(data=net, pool_type='max', kernel=(1, 2), stride=(2, 2))
    net = mx.sym.Convolution(data=net, kernel=(1, 2), num_filter=180)
    net = mx.sym.Activation(data=net, act_type='relu')        
    net = mx.sym.Convolution(data=net, kernel=(1, 1), num_filter=180)
    net = mx.sym.Activation(data=net, act_type='relu')        
    net = mx.sym.flatten(data=net)
    net = mx.sym.FullyConnected(data=net, num_hidden=2048)
    net = mx.sym.Activation(data=net, act_type='relu')            
    net = mx.sym.FullyConnected(data=net, num_hidden=2048)
    net = mx.sym.Activation(data=net, act_type='relu')
    net = mx.sym.FullyConnected(data=net, num_hidden=output_num)    
    net = mx.sym.SoftmaxOutput(data=net, name='softmax')
    #a = mx.viz.plot_network(net)
    #a.render('cnn.net')
    return net
