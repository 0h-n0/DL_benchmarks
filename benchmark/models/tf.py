
"""
tensorflow trainers and models
"""
import os
from functools import partial

from tqdm import tqdm
import tensorflow as tf

from benchmark.models.base_trainer import BaseTrainer


class Trainer(BaseTrainer):
    def __init__(self, model, ngpu, options,
                 data_options=None, time_options=None):
        self.model = model
        self.ngpu = ngpu
        self.gpu_mode = True if ngpu >= 1 else False
        self.time_options = time_options

        if self.gpu_mode:
            pass
            
        if options['benchmark_mode']:
            pass
            
    def set_optimizer(self, opt_type, opt_conf):
        if opt_type == 'SGD':
            with tf.name_scope('momentumSGD_optimizer'):            
                self.optimizer = tf.train.MomentumOptimizer(
                    learning_rate=opt_conf['lr'],
                    momentum=opt_conf['momentum'])            
        else:
            raise NotImplementedError
        
    def run(self, iterator, mode='train'):
        report = dict()
        
        with tf.name_scope('loss'):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                labels=self.model.y_, logits=self.model())
        cross_entropy = tf.reduce_mean(cross_entropy)

        time_series = []
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        total_s = time.perf_counter()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for idx, (x, t) in enumerate(iterator):
                if self.time_options == 'total':            
                    start_event.record()
                    
                self.optimizer.run(feed_dict={x:x, y_:t})
                
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
        self.x = tf.placeholder(tf.float32, [None, channel * xdim * ydim)
        self.y_ = tf.placeholder(tf.float32, [None, output_num])

        self.cnn = partial(cnn,
                           channel=channel,
                           xdim=xdim,
                           ydim=ydim,
                           output_num=output_num)

    def __call__(self):
        return self.cnn(self.x)
    

def cnn(x, channel, xdim, ydim, output_num):    
  with tf.name_scope('reshape'):
    x_image = tf.reshape(x, [-1, 28, 28, 1])

  with tf.name_scope('conv1'):
    W_conv1 = weight_variable([xdim, 3, channel, 180])
    b_conv1 = bias_variable([180])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

  with tf.name_scope('conv2'):
    W_conv2 = weight_variable([1, 3, 1, 180])
    b_conv2 = bias_variable([180])
    h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)
    
  with tf.name_scope('pool1'):
    h_pool1 = max_pool(h_conv1, (1, 2), stride=(2, 2))

  with tf.name_scope('conv3'):
    W_conv3 = weight_variable([1, 3, 1, 180])
    b_conv3 = bias_variable([180])
    h_conv3 = tf.nn.relu(conv2d(h_pool1, W_conv3) + b_conv3)

  with tf.name_scope('conv4'):
    W_conv4 = weight_variable([1, 3, 1, 180])
    b_conv4 = bias_variable([180])
    h_conv4 = tf.nn.relu(conv2d(h_conv3, W_conv4) + b_conv4)

  with tf.name_scope('pool2'):
    h_pool2 = max_pool(h_conv4, (1, 2), stride=(2, 2))
    
  with tf.name_scope('conv5'):
    W_conv5 = weight_variable([1, 2, 1, 180])
    b_conv5 = bias_variable([180])
    h_conv5 = tf.nn.relu(conv2d(h_pool1, W_conv3) + b_conv3)

  with tf.name_scope('conv6'):
    W_conv6 = weight_variable([1, 1, 1, 180])
    b_conv6 = bias_variable([180])
    h_conv6 = tf.nn.relu(conv2d(h_conv3, W_conv4) + b_conv4)
    
  with tf.name_scope('fc1'):
    W_fc1 = weight_variable([3 * 180, 2048])
    b_fc1 = bias_variable([2048])
    h_conv6_flat = tf.reshape(h_pool2, [-1, 3 * 180])
    h_fc1 = tf.nn.relu(tf.matmul(h_conv6_flat, W_fc1) + b_fc1)
    
  with tf.name_scope('fc2'):
    W_fc2 = weight_variable([3 * 180, 2048])
    b_fc2 = bias_variable([2048])
    h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)
    
  with tf.name_scope('fc3'):
    W_fc3 = weight_variable([2048, output_num])
    b_fc3 = bias_variable([output_num])
    y_conv = tf.matmul(h_fc2, W_fc3) + b_fc3
    
  return y_conv

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool(x, kernel=(1, 1), stride=(1, 1)):
  return tf.nn.max_pool(x, ksize=[1, kernel[0], kernel[1], 1],
                        strides=[1, stride[0], stride[1], 1], padding='SAME')

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)
