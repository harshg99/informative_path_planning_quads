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
    def __init__(self,id,model,env,args_dict,imitation_settings = None):
        self.model = model
        self.env = env
        self.ID = id
        self.gradient = []
        self.train_buffer = []
        self.episode_data = []
        self.args_dict = args_dict
        if imitation_settings is None:
            self.imitation_worker = False
        else:
            self.imitation_worker = True
            self.imitation_settings = imitation_settings

    def init_expert(self,imitation_settings):
        pass

    def reset(self,episodeNum):
        # Resets to episode number
        self.episodeNum = episodeNum
        self.env.reset(episode_num = episodeNum)
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
        train_buffer['prev_actions'] = []
        train_buffer['rewards'] = []
        train_buffer['next_obs'] = []
        train_buffer['values'] = []
        train_buffer['valids'] = []
        train_buffer['dones'] = []
        train_buffer['policy'] = []

        if self.args_dict['LSTM']:
            train_buffer['hidden_in'] = []
            train_buffer['hidden_out'] = []
            hidden_in = None
            hidden_out = None

        train_buffer['episode_length'] = self.env.episode_length
        episode_reward = 0
        control_cost = 0
        if RENDER_TRAINING:
            frames = []

        while ((not self.args_dict['FIXED_BUDGET'] and episode_step < self.env.episode_length) \
               or (self.args_dict['FIXED_BUDGET'])):
            train_buffer['obs'].append(observation)
            #print(observation)
            if self.args_dict['LSTM']:
                policy,value,hidden_out = self.model.forward_step(observation,hidden_in)
                train_buffer['hidden_in'].append(hidden_in)
                train_buffer['hidden_out'].append(hidden_out.cpu().detach().numpy())
                hidden_in = hidden_out.cpu().detach().numpy()
            else:
                policy, value = self.model.forward_step(observation)

            policy = policy.cpu().detach().numpy()
            value = value.cpu().detach().numpy()

            action_dict = Utilities.sample_actions(policy)
            train_buffer['actions'].append([action_dict[k] for k in action_dict.keys()])
            train_buffer['values'].append(value[0])
            train_buffer['policy'].append(policy[0])

            rewards,done = self.env.step_all(action_dict)

            train_buffer['rewards'].append(rewards)
            train_buffer['dones'].append(int(done))
            train_buffer['valids'].append(observation['valids'])

            observation = self.env.get_obs_all()
            train_buffer['next_obs'].append(observation)


            episode_step+=1
            episode_reward += np.array(rewards).sum()
            if RENDER_TRAINING and episodeNum%RENDER_TRAINING_WINDOW==0:
                frames += self.env.render(mode='rgb_array')

            if done:
                break

        if RENDER_TRAINING and episodeNum%RENDER_TRAINING_WINDOW==0:
            make_gif(np.array(frames),
                     '{}/episode_{:d}_{:d}_{:.1f}.gif'.format(GIFS_PATH, episodeNum, 0, episode_reward))

        if self.args_dict['LSTM']:
            policy_, value_,_ = self.model.forward_step(observation,hidden_in)
        else:
            policy_, value_ = self.model.forward_step(observation)
        train_buffer['bootstrap_value'] = value_.cpu().detach().numpy()[0]

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

    def compute_il_grads(self,train_buffer):
        pass

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

    def consolidate_buffer(self,buffer):
        # buffer_consolidated = {}
        # for keys in buffer.keys():
        #     if  keys =='obs' or keys=='next_obs':
        #         temp_dict = {}
        #         for keys in
        return buffer

    def work(self,currEpisode):
        if TRAINING_TYPE==TRAINING_OPTIONS.singleThreaded:
            train_buffer,episode_reward,control_cost,episode_step = self.single_threaded_episode(currEpisode)
            if self.args_dict['ALG_TYPE']=='SAC' or self.args_dict['ALG_TYPE']=='SACLSTM':
                self.train_buffer = self.consolidate_buffer(train_buffer)
            else:
                self.train_buffer = train_buffer
        else:
            pass

        if JOB_TYPE==JOB_TYPES.getGradient:
            metrics = self.compute_grads(train_buffer=self.train_buffer)
            metrics_perf = self.env.get_final_metrics()
            metrics_perf['Reward'] = episode_reward
            metrics_perf['Length'] = episode_step
            self.episode_data = {'Losses': metrics, 'Perf': metrics_perf}
        else:
            self.episode_data = {'Perf': {'Reward': episode_reward, 'Control Cost': control_cost,
                                                             'Length': episode_step}}

        return

