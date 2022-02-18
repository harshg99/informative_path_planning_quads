import numpy
import torch

import Utilities
import numpy as np
import time
import threading
from params import *
import Utilities

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
        for j in range(NUM_AGENTS):
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
        episode_reward = 0
        control_cost = 0
        while(episode_step <EPISODE_LENGTH):
            train_buffer['obs'].append(observation)
            policy,value = self.model.forward(torch.tensor(observation,dtype=torch.float32))
            policy = policy.detach().numpy()
            value = value.detach().numpy()

            action_dict = Utilities.get_sampled_actions(policy)
            train_buffer['actions'].append([action_dict[k] for k in action_dict.keys()])
            train_buffer['values'].append(value[0])
            rewards = self.env.step_all(action_dict)
            train_buffer['rewards'].append(rewards)
            observation = self.env.get_obs_all()
            train_buffer['next_obs'] = observation
            episode_step+=1
            episode_reward += np.array(rewards).sum()

            if RENDER_TRAINING:
                self.env.render(mode='rgb_array')


        policy_, value_ = self.model.forward(torch.tensor(observation,dtype=torch.float32))
        train_buffer['bootstrap_value'] = value_.detach().numpy()[0]

        print('MetaAgent{} Episode {} Reward {} Control cost {}'.format(self.ID,episodeNum,episode_reward,control_cost))
        return train_buffer,episode_reward,control_cost

    def multi_threaded_episode(self):
        pass

    def compute_grads(self,train_buffer):
        train_buffer_adv = self.model.get_advantages(train_buffer)
        metric,gradients = self.model.backward(train_buffer_adv)
        self.gradient.append(gradients)
        return metric

    def work(self,currEpisode):
        if TRAINING_TYPE==TRAINING_OPTIONS.singleThreaded:
            self.train_buffer,episode_reward,control_cost = self.single_threaded_episode(currEpisode)
        else:
            pass

        if JOB_TYPE==JOB_TYPES.getGradient:
            metrics = self.compute_grads(train_buffer=self.train_buffer)
        else:
            pass
        self.episode_data = {'Losses':metrics,'Perf':{'Reward':episode_reward,'Control Cost':control_cost}}
        return

