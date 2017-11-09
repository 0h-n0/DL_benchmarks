"""
CNTK models
"""

import os 
import cntk as C
from cntk.device import try_set_default_device, gpu, all_devices
from ctmodel import cnn
from cntk.io import MinibatchSource, CTFDeserializer, StreamDef, StreamDefs, INFINITELY_REPEAT
from cntk.initializer import he_normal
from cntk.layers import AveragePooling, BatchNormalization, Convolution, Dense
from cntk.ops import element_times, relu

input_dim = 3 * 19 * 40    
input_var = C.ops.input_variable(SHAPE, np.float32)
label_var = C.ops.input_variable(LABELSIZE, np.float32)
## declares symbol (input_var, label_var).

z = cnn(input_var, 3000)

ce = C.losses.cross_entropy_with_softmax(z, label_var)
pe = C.metrics.classification_error(z, label_var)
progress_printer = C.logging.ProgressPrinter(tag='Training', num_epochs=niterate)
learning_rate = 0.01
lr_schedule = C.learning_rate_schedule(learning_rate, C.UnitType.minibatch)

momentum = 0.9
m_schedule = C.momentum_schedule(momentum, C.UnitType.minibatch)    
learner = C.learners.momentum_sgd(z.parameters, lr_schedule, m_schedule)

if is_parallel:
    distributed_learner = C.data_parallel_distributed_learner(learner=learner,
                                                              num_quantization_bits=32,
                                                              distributed_after=0)
if is_parallel:
    trainer = C.Trainer(z, (ce, pe), distributed_learner, progress_printer)              
else:
    trainer = C.Trainer(z, (ce, pe), learner, progress_printer)        

aggregate_loss = 0.0
np.random.seed(0)

for i in range(niterate):
    s = time.time()        
    data, _, targets = generator(batch, labelsize, dtype='l')
    trainer.train_minibatch({input_var : data, label_var : targets})
    sample_count = trainer.previous_minibatch_sample_count
    aggregate_loss += trainer.previous_minibatch_loss_average * sample_count
    print(time.time() - s)

if is_parallel:
    distributed.Communicator.finalize() 



class Trainer(object):
    def __init__(self, model, ngpu):
        self.model = model
        self.ngpu = ngpu
        self.gpu_mode = True if ngpu >= 1 else False
        if self.gpu_mode:
            gpus = [i for i in range(self.ngpu)]
            self.model = torch.nn.DataParallel(model, device_ids=gpus)
            self.model.cuda()
                        
        #torch.backends.cudnn.benchmark = True

            
    def set_optimizer(self, opt_type, opt_conf):
        if opt_type == 'SGD':
            self.optimizer = optim.SGD(self.model.parameters(),
                                       lr=opt_conf['lr'],
                                       momentum=opt_conf['momentum'])
        elif opt_type == 'Adam':
            self.optimizer = optim.Adam(self.model.parameters(),
                                        lr=opt_conf['lr'])
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
    net = C.layers.Convolution2D((19, 3), 180, activation=C.ops.relu, pad=False, strides=1)(x)
    net = C.layers.Convolution2D((1, 3), 180, activation=C.ops.relu, pad=False)(net)
    net = C.layers.MaxPooling((1, 2), strides=2)(net)
    net = C.layers.Convolution2D((1, 3), 180, activation=C.ops.relu, pad=False)(net)
    net = C.layers.Convolution2D((1, 3), 180, activation=C.ops.relu, pad=False)(net)
    net = C.layers.MaxPooling((1, 2), strides=2)(net)
    net = C.layers.Convolution2D((1, 3), 180, activation=C.ops.relu, pad=False)(net)
    net = C.layers.Convolution2D((1, 3), 180, activation=C.ops.relu, pad=False)(net)
    net = C.layers.Dense(2048)(net)
    net = C.layers.Dense(2048)(net)
    net = C.layers.Dense(nout, activation=None)(net)
    return net
