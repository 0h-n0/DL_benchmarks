import os
import mxnet as mx
from collections import namedtuple

class Trainer(object):
    def __init__(self, model, ngpu):
        self.model = model
        self.ngpu = ngpu
        self.gpu_mode = True if ngpu >= 1 else False
        data_shapes = [('data', (batch, *SHAPE))]
        label_shapes = [('softmax_label', (batch, labelsize))]
        model.bind(data_shapes=data_shapes,
                   label_shapes=label_shapes)
        
        if self.gpu_mode:
            data = mx.sym.var('data')            
            gpus = [mx.gpu(i) for i in range(ngpu)]
            self.model = mx.mod.Module(symbol=cnn(data, labelsize), data_names=['data'],
                                  label_names=['softmax_label'], context=gpus)
            self.model.init_params(initializer=mx.init.Xavier(magnitude=2.))
        #os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'            
    def set_optimizer(self, opt_type, opt_conf):
        if opt_type == 'SGD':
            self.model.init_optimizer(optimizer='sgd', optimizer_params=(('learning_rate',opt_conf['lr']),))
            metric = mx.metric.create('acc')
        else:
            raise NotImplementedError
        
    def run(self, iterator, mode='train'):
        report = dict()
        criterion = torch.nn.CrossEntropyLoss().cuda()
        for idx, (x, t) in enumerate(iterator):
            total_s = time.perf_counter()            
            x = torch.FloatTensor(x)
            t = torch.LongTensor(t) 
            if self.gpu_mode:
                x = x.cuda()
                t = t.cuda()
            x, t = Variable(x), Variable(t)
            forward_s = time.perf_counter()
            x = self.model(x)
            forward_e = time.perf_counter()
            self.optimizer.zero_grad()
            loss = criterion(x, t)
            backward_s = time.perf_counter()
            loss.backward()
            backward_e = time.perf_counter()            
            self.optimizer.step()
            total_e = time.perf_counter()
            report[idx] = dict(
                forward=forward_e - forward_s,
                backward=backward_e - backward_s,
                total=total_e - total_s
            )
        return report

    
        
def cnn(x, nout):
    net = mx.sym.Convolution(data=x, kernel=(19,3), num_filter=180)
    net = mx.sym.Activation(data=net, act_type='relu')    
    net = mx.sym.Convolution(data=net, kernel=(1,3), num_filter=180)
    net = mx.sym.Activation(data=net, act_type='relu')        
    net = mx.sym.Pooling(data=net, pool_type='max', kernel=(1, 2), stride=(1, 2))
    net = mx.sym.Convolution(data=net, kernel=(1,3), num_filter=180)
    net = mx.sym.Activation(data=net, act_type='relu')        
    net = mx.sym.Convolution(data=net, kernel=(1,3), num_filter=180)
    net = mx.sym.Activation(data=net, act_type='relu')        
    net = mx.sym.Pooling(data=net, pool_type='max', kernel=(1, 2), stride=(1, 2))
    net = mx.sym.Convolution(data=net, kernel=(1,3), num_filter=180)
    net = mx.sym.Activation(data=net, act_type='relu')        
    net = mx.sym.Convolution(data=net, kernel=(1,3), num_filter=180)
    net = mx.sym.Activation(data=net, act_type='relu')        
    net = mx.sym.flatten(data=net)
    net = mx.sym.FullyConnected(data=net, num_hidden=2048)
    net = mx.sym.Activation(data=net, act_type='relu')            
    net = mx.sym.FullyConnected(data=net, num_hidden=2048)
    net = mx.sym.Activation(data=net, act_type='relu')
    net = mx.sym.FullyConnected(data=net, num_hidden=nout)    
    net = mx.sym.SoftmaxOutput(data=net, name='softmax')
    #a = mx.viz.plot_network(net)
    #a.render('cnn.net')
    return net
