import sys
import time
import copy
import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Chain

class Iterator(object):
    def __init__(self, data_type, image_shape, sequence_shape, niteration,
    batch_size, label_size, target_type):
        self.data_type = data_type
        self.image_shape = image_shape
        self.sequence_shape = sequence_shape
        self.niteration = niteration
        self.batch_size = batch_size
        self.label_size = label_size
        self.target_type = target_type
        self._i = 0
        
    def __iter__(self):
        return self
    
    def __next__(self):
        if self._i == self.niteration:
            raise StopIteration()
        self._i += 1
        
        if self.data_type == 'image':
            ### data dimension = [batch, channel, height, width]
            dims = np.prod(self.image_shape)
            data = np.random.random(dims * self.batch_size)
            data = data.reshape(self.batch_size, *self.image_shape)
            ### target dimension = [batch]
            _target = np.random.randint(self.label_size, size=self.batch_size)
            if self.target_type == 'one-hot':
                target = np.zeros((self.batch_size, self.label_size))
                target[np.arange(self.batch_size), _target] = 1
            else:
                target = _target
                
        elif self.data_type == 'sequence':
            data = np.random.random()
            
        return (data, target)

    def __len__(self):
        return self.niteration


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
    

if __name__ == "__main__":
    ngpu = int(sys.argv[1])
    channel = 3
    xdim = 28
    ydim = 28
    output_num = 3000
    data_type = 'image'
    data_config = dict(
        image_shape = (3, 28, 28), # (channel, witdth, height)
        sequence_shape = 28, # feature
        niteration = 1000,
        batch_size = int(sys.argv[2]),
        label_size = 3000,
        target_type = None
    )

    iterator = Iterator(data_type, **data_config)
    model = module.CNN(channel, xdim, ydim, output_num)


    # train
    models = [copy.deepcopy(model).to_gpu(i) for i in range(ngpu)]
    optimizer = chainer.optimizers.SGD(lr=0.01)
    optimizer.setup(models[0])    

    x, t = next(iterator)
    x = x.astype(np.float32)
    t = t.astype(np.int32)
    minibatch = len(x) // ngpu        
    x = [chainer.Variable(
        chainer.cuda.to_gpu(x[j*minibatch:(j+1)*minibatch], j))
         for j in range(ngpu)]
    t = [chainer.Variable(
        chainer.cuda.to_gpu(t[j*minibatch:(j+1)*minibatch], j))
         for j in range(ngpu)]

    for i in range(niteration):
        o = [_model(_x) for _model, _x in zip(self.model, x)]
