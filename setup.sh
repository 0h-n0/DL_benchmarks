#!/bin/bash

mkdir downloads
cd downloads
root=$(pwd)

git clone --recursive https://github.com/pytorch/pytorch.git
git clone --recursive https://github.com/tiny-dnn/tiny-dnn.git
git clone --recursive https://github.com/chainer/chainer.git
git clone --recursive https://github.com/tensorflow/tensorflow.git
git clone --recursive https://github.com/NervanaSystems/neon.git
git clone --recursive https://github.com/NervanaSystems/nervanagpu.git
#git clone --recursive https://github.com/Microsoft/CNTK.git
git clone --recursive https://github.com/apache/incubator-mxnet.git
git clone --recursive https://github.com/sony/nnabla.git
git clone --recursive https://github.com/caffe2/caffe2.git
git clone --recursive https://github.com/gluon-api/gluon-api.git
git clone --recursive https://github.com/cupy/cupy.git
wget http://bitbucket.org/eigen/eigen/get/3.3.4.tar.bz2

#conda create --name benchmark pip -y
#source activate benchmark
conda install -y -c anaconda cmake
conda install -y -c conda-forge bzip2
conda install -y -c mpi4py openmpi
conda install -y -c intel mkl 
conda install -y -c intel/label/test mkl
conda install -y -c intel/label/deprecated mkl

# conda install -y -c anaconda gcc_linux-64
# conda install -y -c anaconda gxx_linux-64

(cd pytorch;pip install -r requirements.txt; python setup.py install)
#(cd pytorch;pip install -r requirements.txt; git checkout refs/tags/v0.2.0python setup.py install)
export CFLAGS=-I$root/downloads/pytorch/torch/lib/build/nccl/include
export LDFLAGS=-L$root/downloads/pytorch/torch/lib/build/nccl/lib
export LD_LIBRARY_PATH=$root/downloads/pytorch/torch/lib/build/nccl/lib:$LD_LIBRARY_PATH

pip install protobuf six filelock numpy cython
#(cd chainer; python setup.py install)
#(cd cupy; python setup.py install)

(cd neon; make sysinstall)
#pip install mxnet-cu80==0.11.0
#pip install tensorflow-gpu
#pip install https://cntk.ai/PythonWheel/GPU/cntk-2.2-cp36-cp36m-linux_x86_64.whl
#(cd chainer; python setup.py install)
#(cd pytorch; python setup.py install)
#(cd neon; pip install -r requirements.txt; pip install -r gpu_requirements.txt; python setup.py install)
