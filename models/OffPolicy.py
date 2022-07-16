import numpy as np
import Utilities
from model_setter import model_setter
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ExponentialLR

class DDQN:
    def __init__(self):
        pass

class SAC:
    def __init__(self,env, params_dict, args_dict):
        self.env = env
        self.input_size = env.input_size
        self.action_size = env.action_size
        self.params_dict = params_dict
        self.args_dict = args_dict
        self._policy = model_setter.set_model(self.env,args_dict) # initialise a policy
        self._qnet = model_setter.set_model(self.env,args_dict) #initialise q network
        self._qnet2 = model_setter.set_model(self.env,args_dict)
        self._tarqet_qnet1 = model_setter.set_model(self.env,args_dict)
        self._tarqet_qnet2 = model_setter.set_model(self.env,args_dict)
        self.device = args_dict['DEVICE']
        self.log_alpha = torch.FloatTensor([-2]).to(self.device)
        self.log_alpha.requires_grad = True

        self.target_entropy = 0.98 * -np.log(1 / self.environment.action_space.n)

        self.q1optim = torch.optim.Adam(self._qnet.parameters(),lr=params_dict['LR'],betas=(0.9,0.99))
        self.scheduler1 = ExponentialLR(self.q2optim,gamma=params_dict['DECAY'])
        self.q2optim = torch.optim.Adam(self._qnet2.parameters(),lr=params_dict['LR'],betas=(0.9,0.99))
        self.scheduler2 = ExponentialLR(self.q2optim,gamma=params_dict['DECAY'])
        self.policyoptim = torch.optim.Adam(self._policy.parameters(),lr=params_dict['LR'],betas=(0.9,0.99))
        self.scheduler3 = ExponentialLR(self.policyoptim,gamma=params_dict['DECAY'])
        self.alphaoptim = torch.optim.Adam(self.log_alpha.parameters(), lr=params_dict['LR'], betas=(0.9, 0.99))
        self.scheduler4 = ExponentialLR(self.alphaoptim, gamma=params_dict['DECAY'])

    def buffer_keys_required(self):
        return self._qnet.buffer_keys_required()
    '''
    Return a policy
    '''
    def forward_step(self,input):
        pass

    def reset(self, episodeNum):
        self.episodeNum = episodeNum

    '''
    Computes the loss
    '''
    def compute_loss(self,buffer):
        a_batch = np.array(buffer['actions'])
        dones = np.array(buffer['dones'])
        rewards_tensor = np.array(buffer['reward'])

        # Compute policy loss
        policy,_,valids,valids_net = self._policy.forward_buffer(buffer['obs'])
        _,values,_,_ = self._qnet.forward_bufer(buffer['obs'])
        _,values2,_,_ = self._qnet2.forward_buffer(buffer['obs'])
        #responsible_outputs = policy.gather(-1, a_batch)
        log_probs = torch.log(torch.clamp(policy,1e-8,1))
        min_Q = torch.minimum(values,values2,dim=-1)

        policy_loss = log_probs*(-min_Q + self.log_alpha.exp().detach()*log_probs)

        alpha_loss = -self.log_alpa.exp()*(log_probs+self.target_entropy).detach()

        with torch.no_grad():
            values_target = self._qnet.forward_buffer(buffer['next_obs'])
            values_target2 = self._qnet2.forward_buffer(buffer['next_obs'])
            soft_values = (policy.detach()*(torch.min(values_target,values_target2) - self.alpha * log_probs)).sum(dim=1)

            next_q_values = rewards_tensor + ~dones * self.DISCOUNT_RATE*soft_values

        soft_q_values = values.gather(1, a_batch).squeeze(-1)
        soft_q_values2 = values2.gather(1, a_batch).squeeze(-1)
        critic_square_error = torch.nn.MSELoss(reduction="none")(soft_q_values, next_q_values)
        critic2_square_error = torch.nn.MSELoss(reduction="none")(soft_q_values2, next_q_values)
        weight_update = [min(l1.item(), l2.item()) for l1, l2 in zip(critic_square_error, critic2_square_error)]

        critic_loss = critic_square_error.mean()
        critic2_loss = critic2_square_error.mean()
        return policy_loss,alpha_loss,critic_loss,critic2_loss


    def update_target_weights(self):
        pass

    def scheduler_step(self):
        self.scheduler1.step()
        self.scheduler2.step()
        self.scheduler3.step()
        self.scheduler4.step()

    def backward(self,buffer):

        p_loss,alpha_loss,q1_loss,q2_loss = self.compute_loss(buffer)
        self.policyoptim.zero_grad()
        p_loss.sum().backward()

        self.q1optim.zero_grad()
        q1_loss.sum().backward()

        self.q2optim.zero_grad()
        q2_loss.sum().backward()

        self.alphaoptim.zero_grad()
        alpha_loss.sum().backward()


