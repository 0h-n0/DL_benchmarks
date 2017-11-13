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
* cntk(keras)
* cntk(gluon?)
* caffe(keras)
* caffe2(python2...)
* neon
* tiny-net
* nnabla
* dynet
* theano(keras)

#### Requirements.

See requirements.txt.

### Setup.

I highly recommend using 'miniconda'. It is very easy to install a lot of DL frameworks with it.
Without 'miniconda', you must spend a lot of time to install them.

```bash
$ wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
$ # for linux
$ sh Miniconda3-latest-Linux-x86_64.sh
$ conda create -m benchmark pip
$ source activate benchmark
```

```
$ conda install pytorch torchvision cuda80 -c soumith
$ # Your GPU is grater than 1080ti.
$ # You should install pytorch from source.
$ # conda install -c anaconda cmake
$ # conda install -c conda-forge bzip2
$ pip install mxnet-cu80
$ # conda install -c pjmtdw mxnet-cudnn (cudnn-5~)
$ pip install https://cntk.ai/PythonWheel/GPU/cntk-2.2-cp36-cp36m-linux_x86_64.whl
$ pip install tensorflow-gpu
$ git clone --recursive https://github.com/NervanaSystems/neon.git
$ (cd neon && make sysinstall)
$ pip install chainer cupy
$ conda install -c conda-forge keras
$ pip install cntk
$ conda install -c mpi4py openmpi
```

See setup.sh.


#### build from source
```
$ # pytorch
$ conda install -c anaconda cmake
$ conda install -c conda-forge bzip2
$ git clone --recursive https://github.com/pytorch/pytorch.git
$ cd pytorch; python setup.py install
$
$ # chainer
$ git clone --recursive https://github.com/chainer/chainer.git
$ git clone --recursive https://github.com/chainer/chainer.git
$ cd chainer; python setup.py install
$ cd cupy; python setup.py install
$
$ # mxnet
$ git clone --recursive https://github.com/apache/incubator-mxnet.git
$ conda install -c intel mkl 
$ conda install -c intel/label/test mkl
$ conda install -c intel/label/deprecated mkl
$ conda install -c anaconda openblas 
$ 
$ $ git clone --recursive https://github.com/NervanaSystems/neon.git
$ (cd neon && make sysinstall)
$ pip install chainer cupy
$ conda install -c conda-forge keras
$ pip install cntk
$ conda install -c mpi4py openmpi
```

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
