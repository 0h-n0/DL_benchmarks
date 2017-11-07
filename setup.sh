#!/bin/bash
mkdir venv

virtualenv -p python3 venv

source venv/bin/activate

pip list
#pip install pyyaml ipython jupyter cython cupy graphviz numpy 
exit

git clone --recursive https://github.com/pytorch/pytorch.git
git clone --recursive https://github.com/tiny-dnn/tiny-dnn.git
git clone --recursive https://github.com/chainer/chainer.git
git clone --recursive https://github.com/tensorflow/tensorflow.git
git clone --recursive https://github.com/NervanaSystems/neon.git
git clone --recursive https://github.com/Microsoft/CNTK.git
git clone --recursive https://github.com/apache/incubator-mxnet.git
git clone --recursive https://github.com/sony/nnabla.git
git clone --recursive https://github.com/caffe2/caffe2.git
git clone --recursive https://github.com/gluon-api/gluon-api.git
#git clone --recursive https://github.com/cupy/cupy.git
wget http://bitbucket.org/eigen/eigen/get/3.3.4.tar.bz2


export CFLAGS=-I/home/kono/Softwares-ks/venv-1080ti/Downloads/pytorch/torch/lib/build/nccl/include
export LDFLAGS=-L/home/kono/Softwares-ks/venv-1080ti/Downloads/pytorch/torch/lib/build/nccl/lib
export LD_LIBRARY_PATH=/home/kono/Softwares-ks/venv-1080ti/Downloads/pytorch/torch/lib/build/nccl/lib:$LD_LIBRARY_PATH

pip install mxnet-cu80==0.11.0
pip install tensorflow-gpu
pip install https://cntk.ai/PythonWheel/GPU/cntk-2.2-cp36-cp36m-linux_x86_64.whl
#(cd chainer; python setup.py install)
#(cd pytorch; python setup.py install)
#(cd neon; pip install -r requirements.txt; pip install -r gpu_requirements.txt; python setup.py install)
