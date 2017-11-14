"""
mxnet trainers and models
"""

import os
import time
from functools import partial
#https://neon.nervanasys.com/index.html/models.html
import neon as nn

class Trainer(object):
    def __init__(self, model, ngpu, options=None):
        self.model = model
        self.ngpu = ngpu
        self.gpu_mode = True if ngpu >= 1 else False
        
        if self.gpu_mode:
            self.gpus = [mx.gpu(i) for i in range(ngpu)]
            
        if options['benchmark_mode']:
            os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '1'
        self.progressbar = options['progressbar']
            
    def set_optimizer(self, opt_type, opt_conf):
        if opt_type == 'SGD':
            self.opt_type = 'sgd'
            self.lr = opt_conf['lr']
            self.metric = mx.metric.CrossEntropy()
        else:
            raise NotImplementedError
        
    def run(self, iterator, mode='train'):
        report = dict()

        # setup mxnet module
        data = mx.sym.var('data')
        module = mx.mod.Module(symbol=self.model(data),
                               context=self.gpus,
                               data_names=['data'],
                               label_names=['softmax_label'])

        B = iterator.batch_size
        C, H, W = iterator.image_shape
        data_shape = (B, C, H, W)
        label_shape = (B,)

        # https://mxnet.incubator.apache.org/tutorials/basic/data.html        
        module.bind(data_shapes=zip(['data'], [data_shape]),
                    label_shapes=zip(['softmax_label'], [label_shape]))
                               
        module.init_params(initializer=mx.init.Xavier(magnitude=2.))
        module.init_optimizer(optimizer=self.opt_type,
                              optimizer_params=(('learning_rate', self.lr),))
        self.metric.reset()
        ## end setup
        if self.progressbar:
            iterator = tqdm(iterator)
        
        for idx, (x, t) in enumerate(iterator):
            total_s = time.perf_counter()
            x = [mx.nd.array(x[i, ...].reshape(1, C, H, W)) for i in range(B)]
            t = [mx.nd.array([t[i]]) for i in range(B)]
            batch = mx.io.DataBatch(x, t)

            forward_s = time.perf_counter()           
            module.forward(batch, is_train=True)
            forward_e = time.perf_counter()
            module.update_metric(self.metric, batch.label)
            backward_s = time.perf_counter()
            module.backward()
            backward_e = time.perf_counter()           
            module.update()
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
