from contextlib import contextmanager

class BaseTrainer(object):
    def __init__(self):
        self._elapsed_time = 0
        
    def set_optimizer(self):
        raise NotImplementedError        

    @contextmanager
    def _record(self, start_event, end_event):
        start_event.record()        
        yield
        end_event.record()
        torch.cuda.synchronize()
        self._elapsed_time = start_event.elapsed_time(end_event)/1000

    def run(self, train_iter, test_iter):
        raise NotImplementedError
