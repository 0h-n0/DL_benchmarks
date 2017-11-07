class Iterator(object):
    def __init__(self, data_type, image_shape, sequence_shape, niteration,
                 batch_size, label_size):
        self.data_type = data_type
        self.image_shape = image_shape
        self.sequence_shape = sequence_shape
        self.niteration = niteration
        self.batch_size = batch_size
        self.label_size = label_size
        self._i = 0
        
    def __iter__(self):
        return self
    
    def next(self):
        if self._i == len(self.niteration):
            raise StopIteration()
        self._i += 1
        value = 1
        
        return value
