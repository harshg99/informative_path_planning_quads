import torch
import torch.nn as nn
from params import *
import numpy as np
import Utilities
from models.Vanilla import Vanilla
from torch.optim.lr_scheduler import ExponentialLR
from models.subnets import *
import torch.nn.functional as F
from models.model_setter import model_setter

class AC:
    def __init__(self,env,params_dict,args_dict):
        self.env = env
        self.input_size = env.input_size
        self.action_size = env.action_size
        self.params_dict = params_dict
        self.args_dict = args_dict

        self._model = model_setter.set_model(self.env,args_dict)
        self._model.to(self.args_dict['DEVICE'])

        self.optim = torch.optim.Adam(self._model.parameters(),lr=params_dict['LR'],betas=(0.9,0.99))
        self.scheduler = ExponentialLR(self.optim,gamma=params_dict['DECAY'])

    def forward_step(self,input):
        return self._model.forward_step(input)

    def reset(self,episodeNum):
        self.episodeNum = episodeNum

    def get_advantages(self,train_buffer):
        rewards_plus = np.copy(train_buffer['rewards']).tolist()
        dones = np.array(train_buffer['dones'])

        if self.args_dict['QVALUE']:
            rewards_plus.append(((1 - dones[-1]) * np.max(train_buffer['bootstrap_value'],axis=-1)).tolist())
        else:
            rewards_plus.append(((1 - dones[-1]) * train_buffer['bootstrap_value'][:, 0]).tolist())

        rewards_plus = np.array(rewards_plus).squeeze()
        discount_rewards = Utilities.discount(rewards_plus,self.args_dict['DISCOUNT'])[:-1]

        if self.args_dict['QVALUE']:
            values_plus = (np.stack(train_buffer['values'])*np.stack(train_buffer['policy'])).sum(axis=-1).tolist()
            values_plus.append(((1 - dones[-1]) * np.max(train_buffer['bootstrap_value'],axis=-1)).tolist())
        else:
            values_plus = train_buffer['values']
            values_plus.append((1 - dones[-1])*train_buffer['bootstrap_value'])
        values_plus = np.array(values_plus).squeeze()
        advantages = np.array(train_buffer['rewards']).squeeze() + \
                     self.args_dict['DISCOUNT']*values_plus[1:] - values_plus[:-1]
        advantages = Utilities.discount(advantages,self.args_dict['DISCOUNT'])
        train_buffer['advantages'] = advantages.copy()
        if self.args_dict['LAMBDA_RET']:
            train_buffer['discounted_rewards'] = Utilities.lambda_return(rewards_plus[:-1],\
                                        values_plus,self.args_dict['DISCOUNT'],self.args_dict['LAMBDA'])
        else:
            train_buffer['discounted_rewards'] = np.copy(discount_rewards)

        return train_buffer

    def compute_loss(self,train_buffer):
        advantages = train_buffer['advantages']
        target_v = train_buffer['discounted_rewards']
        a_batch = np.array(train_buffer['actions'])
        dones = np.array(train_buffer['dones'])


        policy,value,valids,valids_net = self.compute_forward_buffer(train_buffer['obs'])

        target_v = torch.tensor(target_v, dtype=torch.float32).to(self.args_dict['DEVICE'])
        a_batch = torch.tensor(a_batch, dtype=torch.int64).to(self.args_dict['DEVICE'])
        advantages = torch.tensor(advantages, dtype=torch.float32).to(self.args_dict['DEVICE'])
        dones = torch.tensor(np.array(dones), dtype=torch.float32).to(self.args_dict['DEVICE'])
        responsible_outputs = policy.gather(-1, a_batch)
        if self.args_dict['QVALUE']:
            v_l = self.params_dict['value_weight'] * torch.square((value.squeeze()*\
                                                F.one_hot(a_batch,self.env.action_size).squeeze()).sum(dim=-1)- target_v)
        else:
            v_l = self.params_dict['value_weight'] * torch.square(value.squeeze() - target_v)
        e_l = -torch.sum(self.params_dict['entropy_weight'] * \
                                   (policy * torch.log(torch.clamp(policy, min=1e-10, max=1.0))),dim=-1)
        p_l = -self.params_dict['policy_weight'] * torch.log(
        torch.clamp(responsible_outputs.squeeze(), min=1e-15, max=1.0)) * advantages.squeeze()



        valid_l1 = -self.params_dict['valids_weight1'] * torch.sum((1 - valids) * \
                                                    torch.log(torch.clip(1 - valids_net, 1e-7, 1)),dim=-1)
        valid_l2 = -self.params_dict['valids_weight2'] * torch.sum(valids * \
                                                        torch.log(torch.clip(valids_net, 1e-7, 1)),dim=-1)
        valid_l = valid_l1 + valid_l2
        return v_l,p_l,e_l,valid_l

    def backward(self,train_buffer):
        self.optim.zero_grad()
        v_l,p_l,e_l,valid_l = self.compute_loss(train_buffer)
        loss = v_l.sum() + p_l.sum() - e_l.sum() + valid_l.sum()

        self.optim.zero_grad()
        with torch.autograd.detect_anomaly():
            loss.sum().backward()
        # self.optimizer.step()
        norm = torch.nn.utils.clip_grad_norm_(self._model.parameters(), 50)
        v_n = torch.linalg.norm(
            torch.stack([torch.linalg.norm(p.detach()).to("cpu") \
                         for p in self._model.parameters()])).detach().numpy().item()

        gradient = []
        for local_param in self._model.parameters():
            gradient.append(local_param.grad)
        g_n = norm.detach().cpu().numpy().item()
        episode_length = train_buffer['episode_length']
        train_metrics = {'Value Loss': v_l.sum().cpu().detach().numpy().item()/episode_length,
                         'Policy Loss': p_l.sum().cpu().detach().numpy().item()/episode_length,
                         'Entropy Loss': e_l.sum().cpu().detach().numpy().item()/episode_length,
                         'Valid Loss': valid_l.sum().cpu().detach().numpy().item() / episode_length,
                         'Grad Norm': g_n, 'Var Norm': v_n}
        return train_metrics, gradient

    def compute_forward_buffer(self,obs_buffer):
        return self._model.forward_buffer(obs_buffer)

    def state_dict(self):
        return self._model.state_dict()

    def share_memory(self):
        self._model.share_memory()

    def load_state_dict(self,weights):
        self._model.load_state_dict(weights)

class PPO(AC):
    def __init__(self, env,params_dict, args_dict):
        super(PPO, self).__init__(env,params_dict,args_dict)

    def compute_loss(self, train_buffer):
        advantages = train_buffer['advantages']
        target_v = train_buffer['discounted_rewards']
        a_batch = np.array(train_buffer['actions'])
        old_policy = np.array(train_buffer['policy'])
        dones = np.array(train_buffer['dones'])

        policy,value,valids,valids_net = self.compute_forward_buffer(train_buffer['obs'])
        target_v = torch.tensor(target_v, dtype=torch.float32).to(self.args_dict['DEVICE'])
        a_batch = torch.tensor(a_batch, dtype=torch.int64).to(self.args_dict['DEVICE'])
        advantages = torch.tensor(advantages, dtype=torch.float32).to(self.args_dict['DEVICE'])
        #advantages = target_v - value.squeeze().detach()
        #advantages = (advantages - advantages.mean(axis=-1)) / (advantages.std(axis=-1) + 1e-8)

        old_policy = torch.tensor(old_policy.squeeze(),dtype=torch.float32).to(self.args_dict['DEVICE'])

        responsible_outputs = policy.gather(-1, a_batch)
        old_responsible_outputs = old_policy.gather(-1,a_batch)
        ratio = (torch.log(torch.clamp(responsible_outputs,1e-10,1)) \
                - torch.log(torch.clamp(old_responsible_outputs,1e-10,1))).exp()
        dones = torch.tensor(np.array(dones), dtype=torch.float32).to(self.args_dict['DEVICE'])

        v_l = (1-dones)*self.params_dict['value_weight'] * torch.square(value.squeeze() - target_v)
        e_l = -(1 - dones) * torch.sum(self.params_dict['entropy_weight'] * \
                                       (policy * torch.log(torch.clamp(policy, min=1e-10, max=1.0))), dim=-1)

        p_l = -(1-dones)*self.params_dict['policy_weight'] * torch.minimum(
        ratio.squeeze() * advantages.squeeze(),
        torch.clamp(ratio.squeeze(),1-self.params_dict['EPS'],1+self.params_dict['EPS'])*advantages.squeeze())

        #valid_l = self.params_dict['valids_weight']* (valids*torch.log(torch.clip(valids_net,1e-7,1))+ (1-valids)*torch.log(torch.clip(1 - valids_net,1e-7,1)))
        valid_l1 = -(1 - dones) * self.params_dict['valids_weight1'] * torch.sum((1 - valids) * \
                                                                                 torch.log(
                                                                                     torch.clip(1 - valids_net, 1e-7,
                                                                                                1)),dim=1)
        valid_l2 = -(1 - dones) * self.params_dict['valids_weight2'] * torch.sum(valids * \
                                                                                 torch.log(
                                                                                     torch.clip(valids_net, 1e-7, 1)),
                                                                                 dim=-1)
        valid_l = valid_l1 + valid_l2
        return v_l, p_l, e_l,valid_l

