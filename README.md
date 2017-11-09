# DL_benchmarks WIP !!!

DL_benchmarks surveys speed of each DeepLearning frameworks with dummy data.
So, Using dummy data, accuracy of results can not be compared. 

### Supported DeepLearning Frameworks.

* tensorflow
* tensorflow(Eagar)
* tensorflow(Keras)
* pytorch
* chainer
* mxnet
* mxnet(gluon)
* cntk
* cntk(Keras)
* caffe(keras)
* caffe2(python2...)
* neon
* tiny-net
* nnabla
* dynet

#### Requirements.

See requirements.txt.

### Setup.

I highly recommend you to use 'miniconda'.

```bash
$ wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
$ # for linux
$ sh Miniconda3-latest-Linux-x86_64.sh
$ conda create -m benchmark pip
$ source activate benchmark
```

```
$ conda install pytorch torchvision cuda80 -c soumith
$ pip install mxnet-cu80
$ pip install https://cntk.ai/PythonWheel/GPU/cntk-2.2-cp36-cp36m-linux_x86_64.whl
$ pip install tensorflow-gpu
$ git clone --recursive https://github.com/NervanaSystems/neon.git
$ (cd neon && make sysinstall)
$ pip install chainer cupy
```

See setup.sh.

#### How to use.

```bash
$ python -m benchmark.main
$ 
$ python -m benchmark.main with framework=chainer
$ # You can change framework.
$
$ python -m benchmark.main with framework=tensorflow data_config.batch_size=100
$ 
$ python -m benchmark.main print_config
$ # You can change configuration like the below code.
$ # If you want to know more details about how to use it,
$ # Plaese check sacred library and its homepage.
```

#### Result.

WIP
