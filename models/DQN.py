import torch
import torch.nn as nn
from params import *
import numpy as np
import Utilities
from models.Vanilla import Vanilla

class DQN(Vanilla):
    def __init__(self,input_size,action_size,hidden_size):
        super(Vanilla,self).__init__()
        self.hidden_sizes = hidden_size
        self.input_size = input_size
        self.action_size = action_size

        self.policy_layers = self.mlp_block(self.input_size,self.hidden_sizes[0],dropout=False)
        for j in range(len(self.hidden_sizes)-1):
            self.policy_layers.extend(self.mlp_block(self.hidden_sizes[j],self.hidden_sizes[j+1]))
        self.policy_layers.extend([nn.Linear(self.hidden_sizes[-1],self.action_size),nn.Softmax(dim=-1)])
        self.policy_net = nn.Sequential(*self.policy_layers)

        self.value_layers = self.mlp_block(self.input_size, self.hidden_sizes[0], dropout=False)
        for j in range(len(self.hidden_sizes) - 1):
            self.value_layers.extend(self.mlp_block(self.hidden_sizes[j], self.hidden_sizes[j + 1]))
        self.value_layers.extend([nn.Linear(self.hidden_sizes[-1], 1)])
        self.value_net = nn.Sequential(*self.value_layers)

        self.optim = torch.optim.Adam(self.parameters(),lr=LR,betas=[0.9,0.99])

    def mlp_block(self,hidden_size,out_size,dropout=True,droputProb = 0.5,activation = nn.ReLU,batchNorm = False):
        layers = [nn.Linear(hidden_size,out_size),
                               activation()]
        if batchNorm:
            layers.append(nn.BatchNorm1d(out_size))
        if dropout:
            layers.append(nn.Dropout(p=droputProb))
        return layers

    def forward(self,input):
        return self.policy_net(input),self.value_net(input)

    def reset(self,episodeNum):
        self.episodeNum = episodeNum

    def get_advantages(self,train_buffer):
        rewards_plus = np.copy(train_buffer['rewards']).tolist()
        rewards_plus.append(train_buffer['bootstrap_value'][:,0].tolist())
        rewards_plus = np.array(rewards_plus).squeeze()
        discount_rewards = Utilities.discount(rewards_plus,DISCOUNT)[:-1]

        values_plus = train_buffer['values']
        values_plus.append(train_buffer['bootstrap_value'])
        values_plus = np.array(values_plus).squeeze()
        advantages = np.array(train_buffer['rewards']).squeeze() + DISCOUNT*values_plus[1:] - values_plus[:-1]

        train_buffer['advantages'] = advantages
        train_buffer['discounted_rewards'] = np.copy(discount_rewards)

        return train_buffer

    def backward(self,train_buffer):
        self.optim.zero_grad()
        advantages = train_buffer['advantages']
        target_v = train_buffer['discounted_rewards']
        a_batch = np.array(train_buffer['actions'])
        obs = torch.tensor(np.array(train_buffer['obs']).squeeze(axis=1),dtype=torch.float32)

        policy, value = self.forward(obs)
        target_v = torch.tensor(target_v,dtype=torch.float32)
        a_batch = torch.tensor(a_batch,dtype=torch.int64)
        advantages = torch.tensor(advantages,dtype=torch.float32)

        responsible_outputs = policy.gather(-1, a_batch.unsqueeze(-1))
        v_l = value_weight * torch.square(value.squeeze() - target_v)
        e_l = -entropy_weight * (policy * torch.log(torch.clamp(policy, min=1e-10, max=1.0)))
        p_l = -policy_weight * torch.log(torch.clamp(responsible_outputs.squeeze(), min=1e-15, max=1.0)) * advantages.squeeze()


        loss = v_l.sum() + p_l.sum() - e_l.sum()
        self.optim.zero_grad()
        loss.sum().backward()
        # self.optimizer.step()
        norm = torch.nn.utils.clip_grad_norm_(self.parameters(), 80)
        v_n = torch.linalg.norm(
            torch.stack([torch.linalg.norm(p.detach()).to("cpu") for p in self.parameters()])).detach().numpy().item()

        gradient = []
        for local_param in self.parameters():
            gradient.append(local_param.grad)
        g_n = norm.detach().cpu().numpy().item()
        train_metrics = {'Value Loss': v_l.sum().cpu().detach().numpy().item()/EPISODE_LENGTH,
                         'Policy Loss': p_l.sum().cpu().detach().numpy().item()/EPISODE_LENGTH,
                         'Entropy Loss': e_l.sum().cpu().detach().numpy().item()/EPISODE_LENGTH,
                         'Grad Norm': g_n, 'Var Norm': v_n}
        return train_metrics, gradient
