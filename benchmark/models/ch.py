import time
import copy
import numpy as np
import chainer
import chainer.functions as F
from chainer.function_hooks import TimerHook
import chainer.links as L
from chainer import Chain


class Trainer(object):
    def __init__(self, model, ngpu, options=None):
        self.ngpu = ngpu
        self.gpu_mode = True if ngpu >= 1 else False
        if self.gpu_mode:
            self.model = [copy.deepcopy(model).to_gpu(i) for i in range(ngpu)]
        else:
            self.model = model
        if options['benchmark_mode']:            
            chainer.using_config('cudnn_deterministic', True)
            
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
        
        total_s = time.perf_counter()
        for idx, (x, t) in enumerate(iterator):
            hook = TimerHook()
            with hook:
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

                o = [_model(_x) for _model, _x in zip(self.model, x)]
                loss = [F.softmax_cross_entropy(_o, _t) for _o, _t in zip(o, t)]
                self.optimizer.target.cleargrads()
                [_model.cleargrads() for _model in self.model]                        
                [(_loss / self.ngpu).backward()
                 for _model, _loss in zip(self.model, loss)]
            
                [self.model[0].addgrads(_model) for _model in self.model]
                self.optimizer.update()
                [_model.copyparams(self.model[0]) for _model in self.model]
            print(hook.total_time())
            hook.print_report()

        total_e = time.perf_counter()

        
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
    
