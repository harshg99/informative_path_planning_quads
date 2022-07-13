import torch.nn as nn
import torch
import numpy as np
from models.Models import *
from models.Vanilla import *
from models.subnets import *
import torch.nn.functional as F
'''
With transformer backbone, observations are tokenised and embedded and compared with motion primitives that 
are also encoded.
'''
class ModelTrans1(Vanilla):
    def __init__(self,env,params_dict,args_dict):
        super(ModelTrans1,self).__init__(env,params_dict,args_dict)
        self.conv_sizes = params_dict['conv_sizes']
        self.env = env
        self.input_size = env.input_size
        self.action_size = env.action_size
        self.num_graph_nodes = env.num_graph_nodes
        self.params_dict = params_dict
        self.args_dict = args_dict

        # initialises the config for transformer backbone
        self.subnet_config()
        self.obstoken_pos_embedding = self.obstoken_pos_embed()

        self._backbone = self.init_backbone()


        self.policy_layers = mlp_block(self.config['embed_size'] +\
                                       self.action_size + self.num_graph_nodes, \
                                        self.config['embed_size'], dropout=False, activation=nn.LeakyReLU)
        # for j in range(2):
        #     self.policy_layers.extend(mlp_block(self.hidden_sizes[-1],\
        #                                              self.hidden_sizes[-1], dropout=False, activation=nn.LeakyReLU))

        self.policy_layers.extend([nn.Linear(self.config['embed_size'],1)])
        self.policy_net = nn.Sequential(*self.policy_layers)
        self.policy_net.to(self.args_dict['DEVICE'])
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()

        self.value_layers = mlp_block(self.config['embed_size']+ \
                                      self.action_size + self.num_graph_nodes, \
                                        self.config['embed_size'], dropout=False, activation=nn.LeakyReLU)
        # for j in range(2):
        #     self.value_layers.extend(mlp_block(self.hidden_sizes[-1],\
        #                                             self.hidden_sizes[-1],dropout=False,activation=nn.LeakyReLU))
        self.value_layers.extend([nn.Linear(self.config['embed_size'], 1)])
        self.value_net = nn.Sequential(*self.value_layers)
        self.value_net.to(self.args_dict['DEVICE'])

    def subnet_config(self):
        self.config = self.params_dict
        self.config['action_size'] = self.action_size
        self.config['patches'] = int(self.input_size[0]*self.input_size[1]\
                                 /np.square(np.power(2,len(self.conv_sizes)))*len(self.env.scale))
        self.config['input_size'] = self.params_dict['conv_sizes'][-1] + self.config['pos_embed_bits']
        self.config['output_tokens'] = self.action_size
        self.config['out_token_size'] = self.env.motionprim_tokensize
        self.config['DEVICE'] = self.args_dict['DEVICE']

    # Defines the transformer backbone architecture
    def init_backbone(self):
        # Different conv layers for different scales
        size = self.input_size
        size[-1] = int(size[-1]/len(self.env.scale))
        self.conv_blocks = [ConvEncLayer(self.conv_sizes,size\
                                      ,self.args_dict['DEVICE']) for _ in self.env.scale]
        self.Encoder = Encoder(self.config)
        self.Encoder.to(self.args_dict['DEVICE'])
        self.Decoder = Decoder(self.config)
        self.Decoder.to(self.args_dict['DEVICE'])
        self.query_embed = nn.Linear(self.env.motionprim_tokensize,self.config['embed_size'],bias=False)

    def tokenise_obs(self,conv_outputs):
        '''
        conv_outputs: List of outputs for each scale
        @return: modiifed tokens with pos embeds
        '''

        pos_embed = torch.tensor(self.obstoken_pos_embedding,dtype=torch.float32,device=conv_outputs[-1].device)\
                        .unsqueeze(axis=0).unsqueeze(axis=0).repeat(conv_outputs[-1].shape[0],conv_outputs[-1].shape[1],1,1)

        tokens= torch.stack(conv_outputs,dim=-1).reshape((conv_outputs[-1].shape[0],conv_outputs[-1].shape[1],\
                                                          -1,conv_outputs[-1].shape[-1]))

        tokens_embedded = torch.cat([tokens,pos_embed],axis=-1)
        return tokens_embedded

    def obstoken_pos_embed(self):
        '''
        @return: returns a vector with token position embed
        '''
        max_scale = self.env.scale[-1]*self.args_dict['RANGE']*2
        min_patch_scale = self.env.scale[0]*self.args_dict['RANGE']*2/np.sqrt(self.config['token_length'])

        min_rep = max_scale/min_patch_scale
        encodings = []
        for scale in self.env.scale:
            step = int(scale*self.args_dict['RANGE']*2/np.sqrt(self.config['token_length'])/min_patch_scale)
            start = int(min_rep/2 - scale*self.args_dict['RANGE']/(min_patch_scale))
            end = int(min_rep/2 + scale*self.args_dict['RANGE']/(min_patch_scale))
            encoding_x = np.expand_dims(np.expand_dims(np.arange(start,end,step),axis=0).repeat(\
                np.sqrt(self.config['token_length']),axis=0).reshape(-1),axis=-1)
            encoding_y = np.expand_dims(np.expand_dims(np.arange(start, end, step), axis=1).repeat(\
                np.sqrt(self.config['token_length']), axis=1).reshape(-1),axis=-1)
            encoding_x = np.unpackbits(np.uint8(encoding_x),count=int(self.config['pos_embed_bits']/2),axis=1)
            encoding_y = np.unpackbits(np.uint8(encoding_y),count=int(self.config['pos_embed_bits']/2),axis=1)
            enc = np.concatenate((encoding_x,encoding_y),axis = 1)
            encodings.append(enc)
        encodings = np.stack(encodings,axis=0)
        encodings = encodings.reshape((-1,encodings.shape[-1]))
        return encodings

    def get_conv_embeddings(self,input_obs):
        conv_embeddings = []
        B,N,H,W,C = input_obs.shape
        step = int(C/len(self.env.scale))
        for j,layers in enumerate(self.conv_blocks):
            embeddings = layers(input_obs[:,:,:,:,j*step:(j+1)*step])
            conv_embeddings.append(embeddings.permute([0,1,3,4,2]))

        return conv_embeddings

    def forward(self,input,prev_a,graph_node,motion_prims,valid_motion_prims):
        #print(input.shape)
        conv_embedding = self.get_conv_embeddings(input)
        obs_tokens = self.tokenise_obs(conv_embedding)

        B,N,D1,D2 = obs_tokens.shape
        attention_tokens = self.Encoder(obs_tokens.reshape((B*N,D1,D2)))

        query_tokens = self.query_embed(motion_prims.reshape((B*N,motion_prims.shape[-2],motion_prims.shape[-1])))

        decoded_values = self.Decoder(attention_tokens,query_tokens,mask = valid_motion_prims)

        graph_node = graph_node.reshape((B*N,graph_node.shape[-1]))
        prev_a = prev_a.reshape((B * N, prev_a.shape[-1]))
        graph_node_embed = graph_node.unsqueeze(dim=1).repeat(1,query_tokens.shape[1],1)
        prev_a_embed = prev_a.unsqueeze(dim=1).repeat(1,query_tokens.shape[1],1)

        decoded_values_embedded = torch.cat([decoded_values,graph_node_embed,prev_a_embed],dim=-1)
        p_net = self.policy_net(decoded_values_embedded).squeeze(-1)
        valid_motion_prims = valid_motion_prims.reshape((B*N,valid_motion_prims.shape[-1]))
        p_net = p_net.masked_fill(valid_motion_prims==0,-1e8)
        policy = self.softmax(p_net).reshape((B,N,p_net.shape[-1]))
        value = self.value_net(decoded_values_embedded).squeeze(-1)\
            .reshape((B,N,decoded_values_embedded.shape[-2]))
        valids = self.sigmoid(p_net).reshape((B,N,p_net.shape[-1]))

        return policy,value,valids

    def forward_step(self, input):

        obs = input['obs']
        previous_actions = input['previous_actions']
        graph_node = input['node']
        motion_prims = input['mp_embeds']
        valid_motion_prims = input['valids']

        obs = torch.tensor([obs], dtype=torch.float32).to(self.args_dict['DEVICE'])
        prev_a = F.one_hot(torch.tensor([previous_actions]),\
                           num_classes = self.action_size).to(self.args_dict['DEVICE'])
        graph_node = F.one_hot(torch.tensor([graph_node]),\
            num_classes = self.num_graph_nodes).to(self.args_dict['DEVICE'])
        motion_prims = torch.tensor([motion_prims],dtype=torch.float32).to(self.args_dict['DEVICE'])
        valid_motion_prims = torch.tensor([valid_motion_prims],dtype=torch.int64).to(self.args_dict['DEVICE'])
        policy,value,_ = self.forward(obs,prev_a,graph_node,motion_prims,valid_motion_prims)
        return policy,value


    def forward_buffer(self, obs_buffer):
        obs = []
        valids = []
        mp_embeds = []
        prev_a = []
        graph_nodes = []
        for j in obs_buffer:
            obs.append(j['obs'])
            valids.append(j['valids'])
            mp_embeds.append(j['mp_embeds'])
            prev_a.append(j['previous_actions'])
            graph_nodes.append(j['node'])
        obs = torch.tensor(np.array(obs), dtype=torch.float32).to(self.args_dict['DEVICE'])
        prev_a = F.one_hot(torch.tensor(prev_a),\
                           num_classes = self.action_size).to(self.args_dict['DEVICE'])
        graph_nodes = F.one_hot(torch.tensor(np.array(graph_nodes)),
                                num_classes = self.num_graph_nodes).to(self.args_dict['DEVICE'])
        mp_embeds = torch.tensor(np.array(mp_embeds),dtype=torch.float32).to(self.args_dict['DEVICE'])
        valids = torch.tensor(np.array(valids), dtype=torch.int64).to(self.args_dict['DEVICE'])
        policy, value,valids_net = self.forward(obs,prev_a,graph_nodes,mp_embeds,valids)
        valids = torch.tensor(valids,dtype = torch.float32).to(self.args_dict['DEVICE'])
        return policy.squeeze(),value.squeeze(), \
               valids.squeeze(), valids_net.squeeze()

class ModelTrans2(ModelTrans1):
    def __init__(self,env,params_dict,args_dict):
        super(ModelTrans1,self).__init__(env,params_dict,args_dict)
        self.conv_sizes = params_dict['conv_sizes']
        self.env = env
        self.input_size = env.input_size
        self.action_size = env.action_size
        self.num_graph_nodes = env.num_graph_nodes
        self.params_dict = params_dict
        self.args_dict = args_dict

        # initialises the config for transformer backbone
        self.subnet_config()
        self.obstoken_pos_embedding = self.obstoken_pos_embed()

        self._backbone = self.init_backbone()


        self.policy_layers = mlp_block(self.config['embed_size'] +\
                                       self.action_size + self.num_graph_nodes, \
                                        self.config['embed_size'], dropout=False, activation=nn.LeakyReLU)
        # for j in range(2):
        #     self.policy_layers.extend(mlp_block(self.hidden_sizes[-1],\
        #                                              self.hidden_sizes[-1], dropout=False, activation=nn.LeakyReLU))

        self.policy_layers.extend([nn.Linear(self.config['embed_size'],self.action_size)])
        self.policy_net = nn.Sequential(*self.policy_layers)
        self.policy_net.to(self.args_dict['DEVICE'])
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()

        self.value_layers = mlp_block(self.config['embed_size']+ \
                                      self.action_size + self.num_graph_nodes, \
                                        self.config['embed_size'], dropout=False, activation=nn.LeakyReLU)
        # for j in range(2):
        #     self.value_layers.extend(mlp_block(self.hidden_sizes[-1],\
        #                                             self.hidden_sizes[-1],dropout=False,activation=nn.LeakyReLU))
        self.value_layers.extend([nn.Linear(self.config['embed_size'], self.action_size)])
        self.value_net = nn.Sequential(*self.value_layers)
        self.value_net.to(self.args_dict['DEVICE'])

    def init_backbone(self):
        # Different conv layers for different scales
        super().init_backbone()
        self.Encoder = Encoder(self.config)
        self.Encoder.to(self.args_dict['DEVICE'])
    def forward(self,input,prev_a,graph_node,motion_prims,valid_motion_prims):
        #print(input.shape)
        conv_embedding = self.get_conv_embeddings(input)
        obs_tokens = self.tokenise_obs(conv_embedding)

        B,N,D1,D2 = obs_tokens.shape
        attention_tokens = self.Encoder(obs_tokens.reshape((B*N,D1,D2)))

        query_tokens = self.query_embed(motion_prims.reshape((B*N,motion_prims.shape[-2],motion_prims.shape[-1])))
        #decoded_values = self.Decoder(attention_tokens,query_tokens,mask = valid_motion_prims)
        attention_token = attention_tokens[:,0]

        graph_node = graph_node.reshape((B*N,graph_node.shape[-1]))
        prev_a = prev_a.reshape((B * N, prev_a.shape[-1]))
        graph_node_embed = graph_node
        prev_a_embed = prev_a

        decoded_values_embedded = torch.cat([attention_token,graph_node_embed,prev_a_embed],dim=-1)
        p_net = self.policy_net(decoded_values_embedded)
        #valid_motion_prims = valid_motion_prims.reshape((B*N,valid_motion_prims.shape[-1]))
        #p_net = p_net.masked_fill(valid_motion_prims==0,-1e8)
        policy = self.softmax(p_net).reshape((B,N,p_net.shape[-1]))
        value = self.value_net(decoded_values_embedded)
        value = value.reshape((B,N,value.shape[-1]))
        valids = self.sigmoid(p_net).reshape((B,N,p_net.shape[-1]))

        return policy,value,valids

