import torch
import torch.optim as optim
import numpy as np
import ray
import os
import sys
from Worker import Worker
from params import *
from env.env_setter import *
from models.alg_setter import alg_setter
import Utilities
import cProfile
@ray.remote(num_cpus=1,num_gpus=NUM_DEVICES*int(GPU)/(NUM_META_AGENTS+1))
class Runner(object):
    def __init__(self,id,args_dict):
        self.ID= id
        self.env = env_setter.set_env(args_dict)
        self.model =  alg_setter.set_model(self.env,args_dict)
        self.worker = Worker(self.ID,self.model,self.env,args_dict)

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
            jobResults.append([self.worker.train_buffer])
        return jobResults,self.worker.episode_data

    def multiThreadedJob(self):
        pass

if __name__ == '__main__':
    import params as parameters
    params = Utilities.set_dict(parameters)
    #neptune_run = Utilities.setup_neptune(params)

    # Profile the runnner object
    
    rs = Runner(0,params)
    cProfile.run('rs.singleThreadedJob(900)')
