import numpy as np

class Iterator(object):
    def __init__(self, data_type, image_shape, sequence_shape, niteration,
                 batch_size, label_size, target_type, random_generation):
        self.data_type = data_type
        self.image_shape = image_shape
        self.sequence_shape = sequence_shape
        self.niteration = niteration
        self.batch_size = batch_size
        self.label_size = label_size
        self.target_type = target_type
        self.random_generation = random_generation
        self._i = 0
        

        if self.data_type == 'image':
            if not random_generation:                
                ### data dimension = [batch, channel, height, width]
                dims = np.prod(self.image_shape)
                data = np.random.random(dims * self.batch_size)
                self.data = data.reshape(self.batch_size, *self.image_shape)
                ### target dimension = [batch]
                self.target = np.random.randint(self.label_size,
                                                size=self.batch_size)
        elif self.data_type == 'mnist':
            pass
        elif self.data_type == 'cifer-10':
            pass
        elif self.data_type == 'sequence':
            pass
        
            
    def __iter__(self):
        return self
    
    def __next__(self):
        if self._i == self.niteration:
            raise StopIteration()
        self._i += 1

        if not self.random_generation:
            return (self.data, self.target)
        
        if self.data_type == 'image':
            ### data dimension = [batch, channel, height, width]
            dims = np.prod(self.image_shape)
            data = np.random.random(dims * self.batch_size)
            data = data.reshape(self.batch_size, *self.image_shape)
            ### target dimension = [batch]
            _target = np.random.randint(self.label_size, size=self.batch_size)
            if self.target_type == 'one-hot':
                target = np.zeros((self.batch_size, self.label_size))
                target[np.arange(self.batch_size), _target] = 1
            else:
                target = _target
                
        elif self.data_type == 'sequence':
            data = np.random.random()
            
        return (data, target)

    def __len__(self):
        return self.niteration
