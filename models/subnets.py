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
