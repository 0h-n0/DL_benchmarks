"""
CNTK trainers and models
"""

import os

import numpy as np
import cntk as C
from cntk.device import try_set_default_device, gpu, all_devices
from ctmodel import cnn
from cntk.io import MinibatchSource, CTFDeserializer, StreamDef, StreamDefs, INFINITELY_REPEAT
from cntk.initializer import he_normal
from cntk.layers import AveragePooling, BatchNormalization, Convolution, Dense
from cntk.ops import element_times, relu







class Trainer(object):
    def __init__(self, model, ngpu, options=None):
        self.model = model
        self.ngpu = ngpu
        self.gpu_mode = True if ngpu >= 1 else False
        if self.gpu_mode:
            gpus = [i for i in range(self.ngpu)]
            self.is_parallel = False
        if options:
            self.progressbar = options['progressbar']                        

            
    def set_optimizer(self, opt_type, opt_conf):
        if opt_type == 'SGD':
            self.lr_schedule = C.learning_rate_schedule(
                opt_conf['lr'], C.UnitType.minibatch)
            self.m_schedule = C.momentum_schedule(
                opt_conf['momentum'], C.UnitType.minibatch)
        else:
            raise NotImplementedError
        
    def run(self, iterator, mode='train'):
        report = dict()
        input_var = C.ops.input_variable(np.prod(iterator.iamge_shape),
                                         np.float32)
        label_var = C.ops.input_variable(iterator.batch_size, np.float32)
        model = self.model(input_var,)
        ce = C.losses.cross_entropy_with_softmax(model, label_var)
        pe = C.metrics.classification_error(model, label_var)
        z = cnn(input_var)
        learner = C.learners.momentum_sgd(z.parameters, self.lr_schedule, self.m_schedule)
        if self.is_parallel:
            distributed_learner = \
                    C.data_parallel_distributed_learner(learner=learner,
                                                    distributed_after=0)

        progress_printer = \
                    C.logging.ProgressPrinter(tag='Training',
                                              num_epochs=iterator.niteration)            
        if self.is_parallel:
            trainer = C.Trainer(z, (ce, pe), distributed_learner,
                                progress_printer)
        else:
            trainer = C.Trainer(z, (ce, pe), learner, progress_printer)        

        for idx, (x, t) in enumerate(iterator):
            total_s = time.perf_counter()
            trainer.train_minibatch({input_var : x, label_var : t})
            forward_s = time.perf_counter()
            forward_e = time.perf_counter()
            backward_s = time.perf_counter()
            backward_e = time.perf_counter()            
            total_e = time.perf_counter()
            
            report[idx] = dict(
                forward=forward_e - forward_s,
                backward=backward_e - backward_s,
                total=total_e - total_s
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
    net = C.layers.Convolution2D((xdim, 3), 180, activation=C.ops.relu, pad=False, strides=1)(x)
    net = C.layers.Convolution2D((1, 3), 180, activation=C.ops.relu, pad=False)(net)
    net = C.layers.MaxPooling((1, 2), strides=2)(net)
    net = C.layers.Convolution2D((1, 3), 180, activation=C.ops.relu, pad=False)(net)
    net = C.layers.Convolution2D((1, 3), 180, activation=C.ops.relu, pad=False)(net)
    net = C.layers.MaxPooling((1, 2), strides=2)(net)
    net = C.layers.Convolution2D((1, 2), 180, activation=C.ops.relu, pad=False)(net)
    net = C.layers.Convolution2D((1, 1), 180, activation=C.ops.relu, pad=False)(net)
    net = C.layers.Dense(2048)(net)
    net = C.layers.Dense(2048)(net)
    net = C.layers.Dense(output_num, activation=None)(net)
    return net
