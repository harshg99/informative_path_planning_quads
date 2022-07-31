import numpy as np
import Utilities
from models.model_setter import model_setter
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

        self.target_entropy = 0.98 * -np.log(1 / self.env.action_size)

        self.q1optim = torch.optim.Adam(self._qnet.parameters(),lr=params_dict['LR'],betas=(0.9,0.99))
        self.scheduler1 = ExponentialLR(self.q1optim,gamma=params_dict['DECAY'])
        self.q2optim = torch.optim.Adam(self._qnet2.parameters(),lr=params_dict['LR'],betas=(0.9,0.99))
        self.scheduler2 = ExponentialLR(self.q2optim,gamma=params_dict['DECAY'])
        self.policyoptim = torch.optim.Adam(self._policy.parameters(),lr=params_dict['LR'],betas=(0.9,0.99))
        self.scheduler3 = ExponentialLR(self.policyoptim,gamma=params_dict['DECAY'])
        self.alphaoptim = torch.optim.Adam([self.log_alpha], lr=params_dict['LR'], betas=(0.9, 0.99))
        self.scheduler4 = ExponentialLR(self.alphaoptim, gamma=params_dict['DECAY'])
        self.optim = self.policyoptim
        # To indicate to global thread that there are schedulers
        self.scheduler = 1

    def buffer_keys_required(self):
        return self._qnet.buffer_keys_required()
    '''
    Return a policy
    '''
    def forward_step(self,input):
        return self._policy.forward_step(input)

    def reset(self, episodeNum):
        self.episodeNum = episodeNum

    '''
    Computes the loss
    '''
    def compute_loss(self,buffer):
        a_batch = torch.tensor(np.array(buffer['actions']),dtype=torch.int64).to(self.args_dict['DEVICE'])
        dones = torch.tensor(np.array(buffer['dones']),dtype=torch.float32).to(self.args_dict['DEVICE'])
        rewards_tensor = torch.tensor(np.array(buffer['rewards']),dtype=torch.float32).to(self.args_dict['DEVICE'])

        # Compute policy loss
        policy,_,valids,valids_net = self._policy.forward_buffer(buffer['obs'])
        _,values,_,_ = self._qnet.forward_buffer(buffer['obs'])
        _,values2,_,_ = self._qnet2.forward_buffer(buffer['obs'])
        #responsible_outputs = policy.gather(-1, a_batch)
        log_probs = torch.log(torch.clamp(policy,1e-8,1))
        min_Q = torch.minimum(values,values2)

        policy_loss = log_probs*(-min_Q + self.log_alpha.exp().detach()*log_probs)
        alpha_loss = -self.log_alpha.exp()*(log_probs+self.target_entropy).detach()
        valid_l1 = -self.params_dict['valids_weight1'] * torch.sum((1 - valids) * \
                                                    torch.log(torch.clip(1 - valids_net, 1e-7, 1)),dim=-1)
        valid_l2 = -self.params_dict['valids_weight2'] * torch.sum(valids * \
                                                        torch.log(torch.clip(valids_net, 1e-7, 1)),dim=-1)
        valid_l = valid_l1 + valid_l2

        with torch.no_grad():
            _,values_target,_,_ = self._qnet.forward_buffer(buffer['next_obs'])
            _,values_target2,_,_ = self._qnet2.forward_buffer(buffer['next_obs'])
            soft_values = (policy.detach()*(torch.minimum(values_target,values_target2) - self.log_alpha.exp() * log_probs)).sum(dim=1)

            next_q_values = rewards_tensor.squeeze() +\
                            (1-dones) * self.args_dict['DISCOUNT']*soft_values

        soft_q_values = values.gather(1, a_batch).squeeze(-1)
        soft_q_values2 = values2.gather(1, a_batch).squeeze(-1)
        critic_square_error = torch.nn.MSELoss(reduction="none")(soft_q_values, next_q_values)
        critic2_square_error = torch.nn.MSELoss(reduction="none")(soft_q_values2, next_q_values)
        weight_update = [min(l1.item(), l2.item()) for l1, l2 in zip(critic_square_error, critic2_square_error)]

        critic_loss = critic_square_error.mean()
        critic2_loss = critic2_square_error.mean()

        return policy_loss,alpha_loss,critic_loss,critic2_loss,valid_l

    def temperature_loss(self,buffer):

        # Compute policy loss
        policy, _, valids, valids_net = self._policy.forward_buffer(buffer['obs'])
        log_probs = torch.log(torch.clamp(policy, 1e-8, 1))
        alpha_loss = -self.log_alpha.exp() * (log_probs + self.target_entropy).detach()
        return alpha_loss

    def policy_loss(self,buffer):
        # Compute policy loss
        policy, _, valids, valids_net = self._policy.forward_buffer(buffer['obs'])
        _, values, _, _ = self._qnet.forward_buffer(buffer['obs'])
        _, values2, _, _ = self._qnet2.forward_buffer(buffer['obs'])
        # responsible_outputs = policy.gather(-1, a_batch)
        log_probs = torch.log(torch.clamp(policy, 1e-8, 1))
        min_Q = torch.minimum(values, values2)

        policy_loss = policy * (-min_Q + self.log_alpha.exp().detach() * log_probs)
        valid_l1 = -self.params_dict['valids_weight1'] * torch.sum((1 - valids) * \
                                                                   torch.log(torch.clip(1 - valids_net, 1e-7, 1)),
                                                                   dim=-1)
        valid_l2 = -self.params_dict['valids_weight2'] * torch.sum(valids * \
                                                                   torch.log(torch.clip(valids_net, 1e-7, 1)), dim=-1)
        valid_loss = valid_l1 + valid_l2
        total_policy_loss = policy_loss.sum()+ valid_loss.sum()
        return total_policy_loss,policy_loss,valid_loss

    def critic_loss(self,buffer):
        a_batch = torch.tensor(np.array(buffer['actions']),dtype=torch.int64).to(self.args_dict['DEVICE'])
        dones = torch.tensor(np.array(buffer['dones']),dtype=torch.float32).to(self.args_dict['DEVICE'])
        rewards_tensor = torch.tensor(np.array(buffer['rewards']),dtype=torch.float32).to(self.args_dict['DEVICE'])

        policy, _, valids, valids_net = self._policy.forward_buffer(buffer['obs'])

        with torch.no_grad():
            _, values_target, _, _ = self._qnet.forward_buffer(buffer['next_obs'])
            _, values_target2, _, _ = self._qnet2.forward_buffer(buffer['next_obs'])
            policy, _, _,_ = self._policy.forward_buffer(buffer['next_obs'])
            log_probs = torch.log(torch.clamp(policy, 1e-8, 1))
            soft_values = (policy.detach() * (
                        torch.minimum(values_target, values_target2) - self.log_alpha.exp() * log_probs)).sum(dim=1)

            next_q_values = rewards_tensor.squeeze() + \
                            (1 - dones) * self.args_dict['DISCOUNT'] * soft_values

        _, values, _, _ = self._qnet.forward_buffer(buffer['obs'])
        _, values2, _, _ = self._qnet2.forward_buffer(buffer['obs'])

        soft_q_values = values.gather(1, a_batch).squeeze(-1)
        soft_q_values2 = values2.gather(1, a_batch).squeeze(-1)
        critic_square_error = torch.nn.MSELoss(reduction="none")(soft_q_values, next_q_values)
        critic2_square_error = torch.nn.MSELoss(reduction="none")(soft_q_values2, next_q_values)


        critic_loss = critic_square_error.mean()
        critic2_loss = critic2_square_error.mean()
        return critic_loss,critic2_loss

    def update_target_weights(self):
        pass

    def scheduler_step(self):
        self.scheduler1.step()
        self.scheduler2.step()
        self.scheduler3.step()
        self.scheduler4.step()

    def backward(self,buffer):
        # critic loss update and step
        self.q1optim.zero_grad()
        self.q2optim.zero_grad()
        q1_loss,q2_loss = self.critic_loss(buffer)
        q1_loss.sum().backward()
        q2_loss.sum().backward()
        self.q1optim.step()
        self.q2optim.step()

        #Policy loss update and step
        self.policyoptim.zero_grad()
        total_p_loss,p_loss,valid_l = self.policy_loss(buffer)
        total_p_loss.sum().backward()
        self.policyoptim.step()


        self.alphaoptim.zero_grad()
        alpha_loss = self.temperature_loss(buffer)
        alpha_loss.sum().backward()
        self.alphaoptim.step()

        self.soft_update_target_networks(self.params_dict['TAU'])

        buffkeys = list(buffer.keys())
        episode_length = len(buffer[buffkeys[0]])
        train_metrics = {'Q1 Loss': q1_loss.sum().cpu().detach().numpy().item()/episode_length,
                         'Q2 Loss': q2_loss.sum().cpu().detach().numpy().item()/episode_length,
                         'Total Policy Loss': total_p_loss.sum().cpu().detach().numpy().item() / episode_length,
                         'Policy Loss': p_loss.sum().cpu().detach().numpy().item()/episode_length,
                         'Valid Loss': valid_l.sum().cpu().detach().numpy().item() / episode_length}
        return train_metrics,None

    def soft_update_target_networks(self, tau=0.05):
        self.soft_update(self._tarqet_qnet1, self._qnet, tau)
        self.soft_update(self._tarqet_qnet2, self._qnet2, tau)

    def soft_update(self, target_model, origin_model, tau):
        for target_param, local_param in zip(target_model.parameters(), origin_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1 - tau) * target_param.data)

    def share_memory(self):
        self._qnet.share_memory()
        self._qnet2.share_memory()
        self._policy.share_memory()

    def state_dict(self):
        return self._policy.state_dict()

    def load_state_dict(self,weights):
        self._policy.load_state_dict(weights)



