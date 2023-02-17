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
from buffer.replay_buffer import ReplayBuffer

def apply_gradients(global_model, gradients,device):
    global_model.optim.zero_grad()
    for g, global_param in zip(gradients, global_model._model.parameters()):
        if g is not None:
            if g.device is not device:
                global_param._grad = g.to(device)
            else:
                global_param._grad = g
    global_model.optim.step()

def compute_ppo_grads(global_model,train_buffer):
    train_buffer_adv = global_model.get_advantages(train_buffer)
    metric,gradients = global_model.backward(train_buffer_adv)
    return metric,deepcopy(gradients)

def compute_sac_grads(global_model,replay_buffer,params_dict):
    metrics = {}
    for _ in range(params_dict['SAC_GRAD_ITERATIONS']):
        buffer = replay_buffer.get_next(params_dict['BATCH_SIZE'])
        metric,_  = global_model.backward(buffer)
        for key in metric.keys():
            if not key in metrics.keys():
                metrics[key] = [metric[key]]
            else:
                metrics[key].append(metric[key])

    for key in metrics.keys():
        metrics[key] = np.mean(np.array(metrics[key]))

    return metrics

def compute_vae_reconstruction_loss(global_model,buffer,params_dict):
    for _ in params_dict['VAE_GRAD_ITERATIONS']:
        buffer = replay_buffer.get_next()
        metrics,_  = global_model.backward(buffer)
        global_model.gradient_step()

    return metrics

def init_ray():
    ray.init(num_gpus=int(GPU)*int(NUM_DEVICES))


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
        if params['LOAD_BEST_MODEL'] is None:
            load_path = MODEL_PATH+"/checkpoint.pkl"
        else:
            load_path = MODEL_PATH+"/checkpoint{}.pkl".format(params['LOAD_BEST_MODEL'])
        print('Loading Model')
        checkpoint = torch.load(load_path,map_location=params['DEVICE'])
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
    save_number = 0

    if params['ALG_TYPE'] == 'SAC':
        replay_buffer = ReplayBuffer(global_model.buffer_keys_required(), params['CAPACITY'])

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
                    if params['ALG_TYPE'] =='SAC':
                        replay_buffer.add_batch(train_buffer)
                        if replay_buffer.occupied_capacity>params['MIN_CAPACITY']:
                            metrics['Losses'] = compute_sac_grads(global_model,replay_buffer,params)

                    else:
                        metrics['Losses'],gradients = compute_ppo_grads(global_model,train_buffer)
                        apply_gradients(global_model, gradients, device)

            if global_model.scheduler:
                if params['ALG_TYPE'] =='SAC' and replay_buffer.occupied_capacity > params['MIN_CAPACITY']:
                    global_model.scheduler_step()
                if params['ALG_TYPE'] !='SAC':
                    global_model.scheduler.step()
            elif JOB_TYPE == JOB_TYPES.getExperience:
                pass
            else:
                pass

            if (params['ALG_TYPE']=='SAC' and replay_buffer.occupied_capacity>params['MIN_CAPACITY'])\
                    or params['ALG_TYPE'] is not 'SAC':
                tensorboard_writer.update(metrics, curr_episode, neptune_run)
                # get the updated weights from the global network
            weights = global_model.state_dict()
            if reinit_count > RAY_RESET_EPS:
                if joblist == []:
                    print('REINITIALIZING RAY')
                    ray.shutdown()
                    init_ray()
                    meta_agents = [Runner.remote(i,params) for i in range(NUM_META_AGENTS)]
                    reinitialize = True
                else:
                    reinitialize = False

                if reinitialize:
                    reinit_count = NUM_META_AGENTS
                    joblist, curr_episode = init_jobs(meta_agents, weights, curr_episode)
                print(len(joblist))
            else:
                if curr_episode < MAX_EPISODES:
                    joblist.extend([meta_agents[info].job.remote(weights, curr_episode)])
                    reinit_count += 1
                    curr_episode += 1


            if curr_episode % 5000 == 0:

                save_number = (save_number+1)% params['NUM_SAVE_MODEL']
                print('Saving Model', end='\n')
                checkpoint = {"model_state_dict": global_model.state_dict(),
                              "optimizer_state_dict": global_model.optim.state_dict(),
                              "epoch": curr_episode}
                # path_checkpoint = "./" + model_path + "/checkpoint_{}_episode.pkl".format(curr_episode)
                path_checkpoint =  MODEL_PATH + "/checkpoint{}.pkl".format(save_number)
                torch.save(checkpoint, path_checkpoint)
                print('Saved Model', end='\n')

            print('FINISHED THE ASSIGNED JOB!')
    except KeyboardInterrupt:
        print("Killing Programme")
        for a in meta_agents:
            ray.kill(a)




