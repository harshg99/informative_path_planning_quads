import torch
import torch.optim as optim
import numpy as np
import ray
import os
import sys
from env.searchenv import *
from Worker import Worker
from models.Vanilla import Vanilla
from params import *
from models.model_setter import model_setter
from env.env_setter import *

@ray.remote(num_cpus=1,num_gpus=int(GPU)/(NUM_META_AGENTS+1))
class Runner(object):
    def __init__(self,id):
        self.ID= id
        self.env = env_setter.set_env(ENV_TYPE)
        self.model =  model_setter.set_model(self.env.input_size, self.env.action_size,MODEL_TYPE)
        self.worker = Worker(self.ID,self.model,self.env)

    def job(self,glob_weights,episode_num):
        self.model.load_state_dict(glob_weights)
        if TRAINING_TYPE == TRAINING_OPTIONS.singleThreaded:
            jobresults,metrics = self.singleThreadedJob(episode_num)
        info = self.ID
        return jobresults,metrics,info,self.worker.train_buffer

    def singleThreadedJob(self,episode_num):
        jobResults = []
        self.worker.work(episode_num)
        if JOB_TYPE==JOB_TYPES.getGradient:
            jobResults.append(self.worker.gradient)
        else:
            jobResults.append(self.worker.train_buffer)
        return jobResults,self.worker.episode_data

    def multiThreadedJob(self):
        pass

if __name__ == '__main__':
    rs = Runner(0)
    rs.singleThreadedJob(500)