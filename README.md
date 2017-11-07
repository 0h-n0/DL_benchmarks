# DL_benchmarks WIP !!!

DL_benchmarks surveys simple benchmark on DeepLearning frameworks with dummy data.
So, Using dummy data, accuracy of results can not be compared. 

#### Supported DeepLearning Frameworks.

* tensorflow
* pytorch
* chainer
* mxnet
* cntk
* caffe2
* neon
* tiny-net
* nnabla
* mxnet(gluon)

#### Requirements.

See requirements.txt.

#### Setup.

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
