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

def apply_gradients(global_model, gradients,device):
    global_model.optim.zero_grad()
    for g, global_param in zip(gradients, global_model.parameters()):
        if g.device is not device:
            global_param._grad = g.cpu()
        else:
            global_param._grad = g
    global_model.optim.step()

def init_ray(params_dict):
    ray.init(num_gpus=int(params_dict['GPU']))

def reinit_ray(jobList, meta_agents,params_dict):
    if jobList == []:
        print('REINITIALIZING RAY')
        ray.shutdown()
        ray.init(num_gpus=int(GPU))
        meta_agents = [Runner.remote(i) for i in range(NUM_META_AGENTS)]
        return True, meta_agents
    else:
        return False, meta_agents

def init_jobs(agents, weights, curr_episode):
    job_list = []
    for i, agent in enumerate(agents):
        job_list.append(agent.job.remote(weights, curr_episode))
        curr_episode += 1
    return job_list, curr_episode

if __name__=='__main__':
    # Creating the gloabl model
    dummy_env = SearchEnv(numAgents=1)
    global_model = Vanilla(dummy_env.input_size, len(ACTIONS), HIDDEN_SIZES)
    global_model.share_memory()
    global_summary = SummaryWriter(TRAIN_PATH)
    curr_episode = 0
    device = torch.device('cpu')

    # Making the diectory
    if not os.path.isdir(MODEL_PATH):
        os.makedirs(MODEL_PATH)

    if LOAD_MODEL:
        print('Loading Model')
        checkpoint = torch.load(MODEL_PATH+"/checkpoint.pkl")
        global_model.load_state_dict(checkpoint['model_state_dict'])
        global_model.optim.load_state_dict(checkpoint['optimizer_state_dict'])
        curr_episode = checkpoint['epoch']
        print('Model resules at Episode: {}'.format(curr_episode))

    meta_agents = [Runner.remote(i)
                   for i in range(NUM_META_AGENTS)]
    tensorboard_writer = Utilities.Tensorboard(global_summary)
    weights = global_model.state_dict()
    joblist,curr_episode = init_jobs(meta_agents,weights,curr_episode)
    neptune_run = Utilities.setup_neptune()
    reinit_count = 0
    returns, best_return = [], -9999

    try:
        while curr_episode<MAX_EPISODES and joblist!=[]:
            done_id, joblist = ray.wait(joblist)
            # get the results of the task from the object store
            jobResults, metrics, info = ray.get(done_id)[0]

            tensorboard_writer.update(metrics, curr_episode, neptune_run)
            if JOB_TYPE == JOB_TYPES.getGradient:
                # apply gradient on the global network
                for gradients in jobResults[0]:
                    apply_gradients(global_model, gradients, device)

            elif JOB_TYPE == JOB_TYPES.getExperience:
                pass
            else:
                pass

                # get the updated weights from the global network
            weights = global_model.state_dict()
            if reinit_count > RAY_RESET_EPS:
                reinitialize, meta_agents = reinit_ray(joblist, meta_agents)
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




