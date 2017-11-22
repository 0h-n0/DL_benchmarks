from tqdm import tqdm
import sacred
import subprocess as sp

ex = sacred.Experiment()

@ex.config
def config():
    dl_targets = ['torch', 'chainer', 'mxnet', 'tensorflow']
    #dl_targets = ['chainer']    
    maxgpu = 1
    #ngpus = [i for i in range(8, maxgpu+1)]
    ngpus = [1]
    batchs = [i for i in range(100, 2000, 100)] + \
             [i for i in range(2000, 10000, 1000)]
    max_batch_per_gpu = 18000

@ex.automain
def main(dl_targets, ngpus, batchs, max_batch_per_gpu):
    from tqdm import trange
    from time import sleep

    for dl in dl_targets:
        for ngpu in ngpus:
            for batch in batchs:
                cmdline = []
                if batch > ngpu * max_batch_per_gpu:
                    continue
                else:
                    cmdline = "python -m benchmark.main with".split()
                    cmdline += ['framework=' + dl,
                                'ngpu=' + str(ngpu),
                                'batch_size=' + str(batch),
                                "data_options.random_generation=False",
                                "progressbar=False"
                    ]
                    print(" ".join(cmdline))
                    sp.run(cmdline)                    

@ex.command
def clean():
    pass
            


    
