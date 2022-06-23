import ray
import torch
import params
import os

import Utilities
import torch.nn as nn
import torch.optim as optim
import numpy as np
from Runner import Runner
from Worker import Worker
from params import *
from models.Vanilla import Vanilla
from env.searchenv import *
from tensorboardX import SummaryWriter
from models.model_setter import model_setter
from models.alg_setter import alg_setter

from env.env_setter import env_setter
from copy import deepcopy

def apply_gradients(global_model, gradients,device):
    global_model.optim.zero_grad()
    for g, global_param in zip(gradients, global_model._model.parameters()):
        if g.device is not device:
            global_param._grad = g.to(device)
        else:
            global_param._grad = g
    global_model.optim.step()

def compute_ppo_grads(global_model,train_buffer):
    train_buffer_adv = global_model.get_advantages(train_buffer)
    metric,gradients = global_model.backward(train_buffer_adv)
    return metric,deepcopy(gradients)

def init_ray():
    ray.init(num_gpus=int(GPU))


def init_jobs(agents, weights, curr_episode):
    jobs = []
    for i in range(len(agents)):
        jobs.append(agents[i].job.remote(weights, curr_episode))
        curr_episode += 1
    return jobs, curr_episode

if __name__=='__main__':
    # Creating the global model
    import params as parameters
    params = Utilities.set_dict(parameters)
    neptune_run = Utilities.setup_neptune(params)

    dummy_env = env_setter.set_env(params)
    global_model = alg_setter.set_model(dummy_env,params)
    init_ray()
    global_model.share_memory()
    global_summary = SummaryWriter(TRAIN_PATH)
    curr_episode = 0
    device = torch.device(params['DEVICE'])

    # Making the diectory
    if not os.path.isdir(MODEL_PATH):
        os.makedirs(MODEL_PATH)
    if not os.path.isdir(GIFS_PATH):
        os.makedirs(GIFS_PATH)

    if LOAD_MODEL:
        print('Loading Model')
        checkpoint = torch.load(MODEL_PATH+"/checkpoint.pkl")
        global_model.load_state_dict(checkpoint['model_state_dict'])
        global_model.optim.load_state_dict(checkpoint['optimizer_state_dict'])
        curr_episode = checkpoint['epoch']
        print('Model results at Episode: {}'.format(curr_episode))

    meta_agents = [Runner.remote(i,params)
                   for i in range(NUM_META_AGENTS)]
    tensorboard_writer = Utilities.Tensorboard(global_summary)
    weights = global_model.state_dict()
    joblist,curr_episode = init_jobs(meta_agents,weights,curr_episode)

    reinit_count = 0
    returns, best_return = [], -9999

    try:
        while curr_episode<MAX_EPISODES and len(joblist)!=0:

            id, joblist = ray.wait(joblist)

            jobResults, metrics, info,exp = ray.get(id)[0]



            if JOB_TYPE == JOB_TYPES.getGradient:
                # apply gradient on the global network
                for gradients in jobResults[0]:
                    apply_gradients(global_model, gradients, device)
            elif JOB_TYPE == JOB_TYPES.getExperience:
                # apply gradient on the global network
                for train_buffer in jobResults[0]:
                    metrics['Losses'],gradients = compute_ppo_grads(global_model,train_buffer)
                    apply_gradients(global_model, gradients, device)

            if global_model.scheduler:
                global_model.scheduler.step()
            elif JOB_TYPE == JOB_TYPES.getExperience:
                pass
            else:
                pass

            tensorboard_writer.update(metrics, curr_episode, neptune_run)
                # get the updated weights from the global network
            weights = global_model.state_dict()
            if reinit_count > RAY_RESET_EPS:
                if joblist == []:
                    print('REINITIALIZING RAY')
                    ray.shutdown()
                    ray.init(num_gpus=params['NUM_DEVICES']*int(GPU))
                    meta_agents = [Runner.remote(i,params) for i in range(NUM_META_AGENTS)]
                    reinitialize = True
                else:
                    reinitialize = False

                if reinitialize:
                    reinit_count = NUM_META_AGENTS
                    joblist, curr_episode = init_jobs(meta_agents, weights, curr_episode)

            else:
                if curr_episode < MAX_EPISODES:
                    joblist.extend([meta_agents[info].job.remote(weights, curr_episode)])
                    reinit_count += 1
                    curr_episode += 1

            if curr_episode % 100 == 0:
                print('Saving Model', end='\n')
                checkpoint = {"model_state_dict": global_model.state_dict(),
                              "optimizer_state_dict": global_model.optim.state_dict(),
                              "epoch": curr_episode}
                # path_checkpoint = "./" + model_path + "/checkpoint_{}_episode.pkl".format(curr_episode)
                path_checkpoint =  MODEL_PATH + "/checkpoint.pkl"
                torch.save(checkpoint, path_checkpoint)
                print('Saved Model', end='\n')

            print('FINISHED THE ASSIGNED JOB!')
    except KeyboardInterrupt:
        print("Killing Programme")
        for a in meta_agents:
            ray.kill(a)




