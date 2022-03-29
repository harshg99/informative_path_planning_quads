import numpy
import torch

import Utilities
import numpy as np
import time
import threading
from params import *
import Utilities
from env.render import *
from models.model_setter import model_setter

class Worker:
    def __init__(self,id,model,env):
        self.model = model
        self.env = env
        self.ID = id
        self.gradient = []
        self.train_buffer = []
        self.episode_data = []

    def reset(self,episodeNum):
        # Resets to episode number
        self.episodeNum = episodeNum
        self.env.reset()
        self.model.reset(episodeNum)

    def single_threaded_episode(self,episodeNum):
        action_dict = {}
        self.reset(episodeNum)
        for j in range(self.env.numAgents):
            action_dict[j] = 0
        _ = self.env.step_all(action_dict)
        observation = self.env.get_obs_all()
        episode_step = 0

        train_buffer = {}
        train_buffer['obs'] = []
        train_buffer['actions'] = []
        train_buffer['rewards'] = []
        train_buffer['next_obs'] = []
        train_buffer['values'] = []
        train_buffer['valids'] = []
        train_buffer['episode_length'] = self.env.episode_length
        episode_reward = 0
        control_cost = 0
        if RENDER_TRAINING:
            frames = []
        while(episode_step <self.env.episode_length):
            train_buffer['obs'].append(observation)
            #print(observation)
            policy,value = self.model.forward_step(observation)

            policy = policy.detach().numpy()
            value = value.detach().numpy()

            action_dict = Utilities.get_sampled_actions(policy)
            train_buffer['actions'].append([action_dict[k] for k in action_dict.keys()])
            train_buffer['values'].append(value[0])
            rewards,done = self.env.step_all(action_dict)
            train_buffer['rewards'].append(rewards)
            train_buffer['valids'].append(observation['valids'])
            observation = self.env.get_obs_all()
            train_buffer['next_obs'] = observation

            episode_step+=1
            episode_reward += np.array(rewards).sum()
            if RENDER_TRAINING and episodeNum%RENDER_TRAINING_WINDOW==0:
                frames.append(self.env.render(mode='rgb_array'))

            if done:
                break

        if RENDER_TRAINING and episodeNum%RENDER_TRAINING_WINDOW==0:
            make_gif(np.array(frames),
                     '{}/episode_{:d}_{:d}_{:.1f}.gif'.format(GIFS_PATH, episodeNum, 0, episode_reward))

        policy_, value_ = self.model.forward_step(observation)
        train_buffer['bootstrap_value'] = value_.detach().numpy()[0]

        print('MetaAgent{} Episode {} Reward {} Control cost {} Length {}'.format(self.ID,episodeNum,episode_reward,control_cost,episode_step))
        return train_buffer,episode_reward,control_cost,episode_step

    def multi_threaded_episode(self):
        pass

    def compute_grads(self,train_buffer):
        self.gradient = []
        train_buffer_adv = self.model.get_advantages(train_buffer)
        metric,gradients = self.model.backward(train_buffer_adv)
        self.gradient.append(gradients)
        return metric

    def compute_grads_batch(self,train_buffer):
        train_buffer_adv = self.model.get_advantages(train_buffer)
        length = train_buffer_adv['obs'].shape[0]
        self.gradient = []
        for j in range(GRADIENT_TYPE*2):
            buffer = self.sample_batch_buffer(train_buffer_adv,batchSize = int(length/GRADIENT_TYPE))
            metric, gradients = self.model.backward(buffer)
            self.gradient.append(gradients)
        pass

    def sample_batch_buffer(self,train_buffer_adv,batchSize=128):
        buffer = {}
        idxs = np.random.randint(0,train_buffer_adv['obs'].shape[0],batchSize)
        for k,v in train_buffer_adv.items():
            buffer[k] = train_buffer_adv[k][idxs]
        pass

    def work(self,currEpisode):
        if TRAINING_TYPE==TRAINING_OPTIONS.singleThreaded:
            self.train_buffer,episode_reward,control_cost,episode_step = self.single_threaded_episode(currEpisode)
        else:
            pass

        if JOB_TYPE==JOB_TYPES.getGradient:
            metrics = self.compute_grads(train_buffer=self.train_buffer)
        else:
            pass
        self.episode_data = {'Losses':metrics,'Perf':{'Reward':episode_reward,'Control Cost':control_cost, 'Length':episode_step}}
        return

