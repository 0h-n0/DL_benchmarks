import time
import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Chain

class Trainer(object):
    def __init__(self, model, ngpu):
        self.model = model
        self.ngpu = ngpu
        self.gpu_mode = True if ngpu >= 1 else False
        if self.gpu_mode:
            chainer.cuda.get_device_from_id(0).use()
            self.model.to_gpu()
            
    def set_optimizer(self, opt_type, opt_conf):
        if opt_type == 'SGD':
            self.optimizer = chainer.optimizers.SGD(lr=opt_conf['lr'])
            self.optimizer.setup(self.model)
        elif opt_type == 'MomentumSGD':
            self.optimizer = chainer.optimizers.MomentumSGD(lr=opt_conf['lr'],
                                                            momentum=opt_conf['momentum'])
            self.optimizer.setup(self.model)
        elif opt_type == 'Adam':
            self.optimizer = chainer.optimizers.Adam(lr=opt_conf['lr'])
            self.optimizer.setup(self.model)
        else:
            raise NotImplementedError
        self.optimizer.use_cleargrads()
        
    def run(self, iterator, mode='train'):
        report = dict()
        for idx, (x, t) in enumerate(iterator):
            total_s = time.perf_counter()
            x = chainer.Variable(x.astype(np.float32))
            t = chainer.Variable(t.astype(np.int32))
            if self.gpu_mode:
                x.to_gpu()
                t.to_gpu()
            forward_s = time.perf_counter()
            o = self.model(x)
            forward_e = time.perf_counter()
            self.model.cleargrads()
            loss = F.softmax_cross_entropy(o, t)
            backward_s = time.perf_counter()
            loss.backward()
            backward_e = time.perf_counter()            
            self.optimizer.update()
            total_e = time.perf_counter()
            report[idx] = dict(
                forward=forward_e - forward_s,
                backward=backward_e - backward_s,
                total=total_e - total_s
            )
        return report


class CNN(Chain):
    def __init__(self, channel, xdim, ydim, output_num):
        super(CNN, self).__init__(
            conv1=L.Convolution2D(channel, 180, (xdim, 3), stride=1),
            conv1a=L.Convolution2D(180, 180, (1, 3)),
            conv2=L.Convolution2D(180, 180, (1, 3), stride=1),
            conv2a=L.Convolution2D(180, 180, (1, 3)),
            conv3=L.Convolution2D(180, 180, (1, 2), stride=1),
            conv3a=L.Convolution2D(180, 180, (1, 1)),
            fc4=L.Linear(540, 2048),
            fc5=L.Linear(2048, 2048),
            fc6=L.Linear(2048, output_num)
        )

    def __call__(self, x):
        h = self.conv1(x)
        h = F.relu(self.conv1(x))
        h = F.max_pooling_2d(F.relu(self.conv1a(h)), (1, 2), stride=2)
        h = F.relu(self.conv2(h))
        h = F.max_pooling_2d(F.relu(self.conv2a(h)), (1, 2), stride=2)
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv3a(h))
        h = F.relu(self.fc4(h))
        h = F.relu(self.fc5(h))
        y = self.fc6(h)
        return y
