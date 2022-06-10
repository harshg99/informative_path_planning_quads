import numpy as np

class traj_buffer:
    '''
    @params: keys_list:list list containing keynames for buffer
    '''
    def __init__(self,keys_list:list):
        self.buffer = {}
        for keys in keys_list:
            self.buffer[keys] = []
        self.buffer_keys = self.buffer.keys()

    '''
        @params: sample: dict dictionary containing keynames for buffer 
    '''
    def add_sample(self,sample):
        for keys in sample.keys():
            if keys in self.buffer_keys:
                self.buffer[keys].append(sample[keys])
            else:
                self.buffer[keys] = [sample[keys]]
                self.buffer_keys = self.buffer.keys()

    '''
        clears buffer
    '''
    def clear_buffer(self,sample):
        for keys in sample.keys():
            if keys in self.buffer_keys:
                self.buffer[keys].append(sample[keys])
            else:
                self.buffer[keys] = [sample[keys]]
                self.buffer_keys = self.buffer.keys()