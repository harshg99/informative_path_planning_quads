import torch.nn as nn
import torch
import numpy as np

def mlp_block(hidden_size, out_size, dropout=True, droputProb=0.5, activation=None, batchNorm=False):
    if activation is None:
        layers = [nn.Linear(hidden_size, out_size)]
    else:
        layers = [nn.Linear(hidden_size, out_size), activation()]
    if batchNorm:
        layers.append(nn.BatchNorm1d(out_size))
    if dropout:
        layers.append(nn.Dropout(p=droputProb))
    return layers

def conv_block(kernel_size,in_channels,out_channels,stride,\
               dropout=False,droputProb=0.5, activation=None, batchNorm=False,padding = 'same'):
    if activation is None:
        layers = [nn.Conv2d(in_channels,out_channels,kernel_size,stride=stride,padding=padding),
                  nn.Conv2d(out_channels,out_channels,kernel_size,stride=stride,padding=padding)]
    else:
        layers = [nn.Conv2d(in_channels,out_channels,kernel_size,stride=stride,padding=padding),
                  activation(),
                  nn.Conv2d(out_channels,out_channels,kernel_size,stride =stride,padding=padding),
                  activation()]
    if batchNorm:
        layers.append(nn.BatchNorm2d(out_channels))
    if dropout:
        layers.append(nn.Dropout(p=droputProb))

    return nn.Sequential(*layers)

class MLPLayer(nn.Module):
    def __init__(self, hidden_sizes,input_size):
        super(MLPLayer,self).__init__()
        self.hidden_sizes = hidden_sizes
        self.input_size = input_size
        self.layers = mlp_block(self.input_size,\
                                self.hidden_sizes[0], dropout=False)
        for j in range(len(self.hidden_sizes) - 1):
            self.layers.extend(
                mlp_block(self.hidden_sizes[j], self.hidden_sizes[j + 1], \
                          dropout=False, activation=nn.LeakyReLU))

        self.layers = nn.Sequential(*self.layers)

    def forward(self, input):
        input = input.view(input.shape[0],input.shape[1],-1)
        return self.layers(input)

class ConvLayer(nn.Module):
    def __init__(self, hidden_size,input_size):
        super(ConvLayer, self).__init__()
        self.hidden_layers = hidden_size
        self.kernel_size = 3
        self.input_size = input_size
        self.layers = []
        self.layers.append(conv_block(self.kernel_size,self.input_size[-1],\
                                      self.hidden_layers[0],stride=1,\
                                      activation=nn.LeakyReLU))
        for j in range(len(self.hidden_layers)-1):
            self.layers.append(conv_block(self.kernel_size,self.hidden_layers[j],\
                                          self.hidden_layers[j+1],stride=1,\
                                          activation=nn.LeakyReLU))
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2,padding=0)
        self.identitylayers = []
        # self.identitylayers.append(conv_block(1,self.input_size[-1],\
        #                               self.hidden_layers[0],stride=2,\
        #                               activation=nn.LeakyReLU,padding = 0))
        #
        # for j in range(len(self.hidden_layers)-1):
        #     self.identitylayers.append(conv_block(1,self.hidden_layers[j],\
        #                               self.hidden_layers[j+1],stride=2,\
        #                               activation=nn.LeakyReLU,padding = 0))
        self.identitylayers.append(nn.Conv2d(in_channels=self.input_size[-1],\
                                             out_channels=self.hidden_layers[0],\
                                             kernel_size=1,stride=2,padding=0))

        for j in range(len(self.hidden_layers)-1):
            self.identitylayers.append(nn.Conv2d(in_channels=self.hidden_layers[j],\
                                                 out_channels=self.hidden_layers[j+1],\
                                                 kernel_size=1,stride=2,padding=0))
        self.fc = nn.Linear(self.hidden_layers[-1]*4,self.hidden_layers[-1])

    def forward(self, input):

        # shape is (batch,num_agents,H,W,C)
        B,N,H,W,C = input.shape
        intermediate = input.reshape([B*N,H,W,C])
        intermediate = torch.permute(intermediate, [0,3,1,2])
        for layer,downsample in zip(self.layers,self.identitylayers):
            identity = downsample(intermediate)
            intermediate = layer(intermediate)
            intermediate = self.pool(intermediate)
            intermediate+=identity
            intermediate = nn.LeakyReLU()(intermediate)
        #B_,H_,W_,C_ = intermediate.shape
        intermediate = nn.LeakyReLU()(self.fc(intermediate.reshape([B,N,-1])))
        return intermediate


class MHA(nn.Module):
    def __init__(self,config):
        # Number of attention blocks
        super(MHA, self).__init__()
        self.num_heads = config['num_heads']
        self.hidden = config['hidden']
        self.config = config
        # Concat and normalisation
        self.attnblocks = [ScaledDotProductAttention(config) for _ in range(self.num_heads)]
        # Feedforward to compressed state
        self.outlayer = nn.Linear(in_features=self.num_heads*self.hidden,out_features=self.hidden)

    def forward(self,keys_in,query_in=None):
        for j,blocks in enumerate(self.attnblocks):
            if j==0:
                concat_output = blocks(keys_in,query_in).unsqueeze(-1)
            else:
                concat_output = torch.concat([concat_output,blocks(keys_in,query_in).unsqueeze(-1)],dim=-1)
        mhaout = self.outlayer(concat_output.flatten(start_dim=-2,end_dim=-1))
        return mhaout

# scaled dot product attention
class ScaledDotProductAttention(nn.Module):
    def __init__(self,config):
        super(ScaledDotProductAttention, self).__init__()
        self.key_length = config['action_size']
        #self.key_size = config['key_size']
        self.hidden = config['hidden']
        self.embed_size = config['embed']
        self.key_network = nn.Linear(self.embed_size,self.hidden)
        self.value_network = nn.Linear(self.embed_size,self.hidden)
        self.query_network = nn.Linear(self.embed_size,self.hidden)

    def forward(self,keys_in,query_in=None):
        if query_in is None:
            query_in = keys_in
        keys = self.key_network(keys_in)
        query = self.query_network(query_in)
        value = self.value_network(keys_in)
        factor = np.sqrt(self.hidden)
        kT = keys.transpose(-2,-1)
        QK = torch.matmul(query,kT)
        QK = torch.softmax(QK/factor,dim=-1)
        QKV = torch.matmul(QK,value)
        return QKV
