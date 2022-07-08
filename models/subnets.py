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
               dropout=False,droputProb=0.5, activation=None, batchNorm=False,padding = 'same',device='cpu'):
    if activation is None:
        layers = [nn.Conv2d(in_channels,out_channels,kernel_size,stride=stride,padding=padding).to(device),
                  nn.Conv2d(out_channels,out_channels,kernel_size,stride=stride,padding=padding).to(device)]
    else:
        layers = [nn.Conv2d(in_channels,out_channels,kernel_size,stride=stride,padding=padding).to(device),
                  activation(),
                  nn.Conv2d(out_channels,out_channels,kernel_size,stride =stride,padding=padding).to(device),
                  activation()]
    if batchNorm:
        layers.append(nn.BatchNorm2d(out_channels))
    if dropout:
        layers.append(nn.Dropout(p=droputProb))

    return nn.Sequential(*layers)

class MLPLayer(nn.Module):
    def __init__(self, hidden_sizes,input_size,device):
        super(MLPLayer,self).__init__()
        self.hidden_sizes = hidden_sizes
        self.input_size = input_size
        self.device = device
        self.layers = mlp_block(self.input_size,\
                                self.hidden_sizes[0], dropout=False)
        for j in range(len(self.hidden_sizes) - 1):
            self.layers.extend(
                mlp_block(self.hidden_sizes[j], self.hidden_sizes[j + 1], \
                          dropout=False, activation=nn.LeakyReLU))

        self.layers = nn.Sequential(*self.layers).to(self.device)

    def forward(self, input):
        input = input.view(input.shape[0],input.shape[1],-1)
        return self.layers(input)

class ConvLayer(nn.Module):
    def __init__(self, hidden_size,input_size,device):
        super(ConvLayer, self).__init__()
        self.hidden_layers = hidden_size
        self.kernel_size = 3
        self.input_size = input_size
        self.layers = []
        self.device = device
        self.layers.append(conv_block(self.kernel_size,self.input_size[-1],\
                                      self.hidden_layers[0],stride=1,\
                                      activation=nn.LeakyReLU,device=self.device))
        for j in range(len(self.hidden_layers)-1):
            self.layers.append(conv_block(self.kernel_size,self.hidden_layers[j],\
                                          self.hidden_layers[j+1],stride=1,\
                                          activation=nn.LeakyReLU,device=self.device))
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2,padding=0).to(self.device)
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
                                             kernel_size=1,stride=2,padding=0).to(self.device))

        for j in range(len(self.hidden_layers)-1):
            self.identitylayers.append(nn.Conv2d(in_channels=self.hidden_layers[j],\
                                                 out_channels=self.hidden_layers[j+1],\
                                                 kernel_size=1,stride=2,padding=0).to(self.device))
        self.fc = nn.Linear(self.hidden_layers[-1]*4,self.hidden_layers[-1]).to(self.device)

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

class ConvEncLayer(nn.Module):
    def __init__(self, hidden_size,input_size,device):
        super(ConvEncLayer, self).__init__()
        self.hidden_layers = hidden_size
        self.kernel_size = 3
        self.input_size = input_size
        self.layers = []
        self.device = device
        self.relu = nn.LeakyReLU()
        self.layers.append(conv_block(self.kernel_size,self.input_size[-1],\
                                      self.hidden_layers[0],stride=1,\
                                      activation=nn.LeakyReLU,device=self.device))
        for j in range(len(self.hidden_layers)-1):
            self.layers.append(conv_block(self.kernel_size,self.hidden_layers[j],\
                                          self.hidden_layers[j+1],stride=1,\
                                          activation=nn.LeakyReLU,device=self.device))
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2,padding=0).to(self.device)
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
                                             kernel_size=1,stride=1,padding=0).to(self.device))

        for j in range(len(self.hidden_layers)-1):
            self.identitylayers.append(nn.Conv2d(in_channels=self.hidden_layers[j],\
                                                 out_channels=self.hidden_layers[j+1],\
                                                 kernel_size=1,stride=1,padding=0).to(self.device))

    def forward(self, input):

        # shape is (batch,num_agents,H,W,C)
        B,N,H,W,C = input.shape
        intermediate = input.reshape([B*N,H,W,C])
        intermediate = torch.permute(intermediate, [0,3,1,2])
        for layer,downsample in zip(self.layers,self.identitylayers):
            identity = downsample(intermediate)
            intermediate = layer(intermediate)
            intermediate += identity
            intermediate = self.pool(intermediate)
            intermediate = self.relu(intermediate)
        B_,C_,H_,W_ = intermediate.shape
        intermediate = intermediate.reshape([B, N, C_,H_,W_])
        return intermediate

class MLPEncLayer(nn.Module):
    def __init__(self, hidden_size,input_size,device):
        super(MLPEncLayer, self).__init__()
        self.hidden_layers = hidden_size
        self.kernel_size = 3
        self.input_size = input_size
        self.layers = []
        self.device = device
        self.relu = nn.ReLU
        self.patch_size = np.power(2,len(self.hidden_layers))

        self.layers.append(conv_block(self.kernel_size,self.input_size[-1],\
                                      self.hidden_layers[0],stride=1,\
                                      activation=nn.LeakyReLU,device=self.device))
        for j in range(len(self.hidden_layers)-1):
            self.layers.append(conv_block(self.kernel_size,self.hidden_layers[j],\
                                          self.hidden_layers[j+1],stride=1,\
                                          activation=nn.LeakyReLU,device=self.device))
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2,padding=0).to(self.device)
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
                                             kernel_size=1,stride=2,padding=0).to(self.device))

        for j in range(len(self.hidden_layers)-1):
            self.identitylayers.append(nn.Conv2d(in_channels=self.hidden_layers[j],\
                                                 out_channels=self.hidden_layers[j+1],\
                                                 kernel_size=1,stride=2,padding=0).to(self.device))

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
            intermediate = nn.ReLU()(intermediate)
        B_,C_,H_,W_ = intermediate.shape
        intermediate = intermediate.reshape([B, N, C_,H_,W_])
        return intermediate

class MHA(nn.Module):
    def __init__(self,config):
        # Number of attention blocks
        super(MHA, self).__init__()
        self.num_heads = config['num_heads']
        self.embed_size = config['embed_size']
        self.config = config
        self.single_head_size = int(self.embed_size/self.num_heads)
        config['head_embed_size'] = self.single_head_size
        # Concat and normalisation
        self.attnblocks = [ScaledDotProductAttention(config).to(self.config['DEVICE']) for _ in range(self.num_heads)]
        # Feedforward to compressed state


    def forward(self,query_in,keys_in,value_in,mask = None):
        for j,blocks in enumerate(self.attnblocks):
            if j==0:
                concat_output = blocks(keys_in,query_in,value_in,mask).unsqueeze(-1)
            else:
                concat_output = torch.concat([concat_output,blocks(keys_in,query_in,value_in,mask)\
                                             .unsqueeze(-1)],dim=-1)
        return concat_output.view(concat_output.shape[0],concat_output.shape[1],-1)

# scaled dot product attention
class ScaledDotProductAttention(nn.Module):
    def __init__(self,config):
        super(ScaledDotProductAttention, self).__init__()
        self.config = config
        self.key_length = config['action_size']
        self.embed_in_size = config['embed_size']
        #self.key_size = config['key_size']
        self.embed_out_size = config['head_embed_size']
        self.key_network = nn.Linear(self.embed_in_size,self.embed_out_size,bias=False).to(self.config['DEVICE'])
        self.value_network = nn.Linear(self.embed_in_size,self.embed_out_size,bias=False).to(self.config['DEVICE'])
        self.query_network = nn.Linear(self.embed_in_size,self.embed_out_size,bias=False).to(self.config['DEVICE'])

    def forwartd(self,keys_in,query_in,value_in,mask = None):
        if query_in is None:
            query_in = keys_in
        keys = self.key_network(keys_in)
        query = self.query_network(query_in)
        value = self.value_network(value_in)
        factor = np.sqrt(self.embed_out_size)
        kT = keys.transpose(-2,-1)
        QK = torch.matmul(query,kT)

        if mask is not None:
            assert mask.size[1] == QK.size[1]
            mask = mask.repeat([QK.size[0],QK.size[1],QK.size[2]])
            QK.masked_fill(mask,-1e8)

        QK = torch.softmax(QK/factor,dim=-1)
        QKV = torch.matmul(QK,value)
        return QKV

class LayerNormalisation(nn.Module):
    def __init__(self, embed_dim):
        super(LayerNormalisation, self).__init__()
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        return self.norm(x.reshape(-1, x.size(-1))).reshape(x.shape)

class EncoderLayer(nn.Module):
    def __init__(self,config):
        super(EncoderLayer,self).__init__()
        self.config = config
        # self.key_size = config['key_size']
        self.num_heads = config['num_heads']
        self.input_size = config['input_size']
        self.embed_size = config['embed_size']
        #self.mhablock = MHA(config).to(self.config['DEVICE'])
        self.mhablock = nn.MultiheadAttention(self.embed_size,self.num_heads,\
                                              batch_first=True).to(self.config['DEVICE'])
        self.layernorm = LayerNormalisation(self.embed_size).to(self.config['DEVICE'])
        self.fclayers = [nn.Linear(self.num_heads*int(self.embed_size/self.num_heads),self.embed_size),\
                         nn.ReLU(),nn.Linear(self.embed_size,self.embed_size)]
        self.fc = nn.Sequential(*self.fclayers).to(self.config['DEVICE'])

    def forward(self,input,mask=None):
        x,_ = self.mhablock(input,input,input)
        x += input
        x = self.layernorm(x)
        x0 = self.fc(x)
        x0 += x
        x0 = self.layernorm(x0)
        return x0

class DecoderLayer(nn.Module):
    def __init__(self,config):
        super(DecoderLayer,self).__init__()
        self.config = config
        # self.key_size = config['key_size']
        self.num_heads = config['num_heads']
        self.input_size = config['input_size']
        self.embed_size = config['embed_size']
        #self.mhablock = MHA(config).to(self.config['DEVICE'])
        self.mhablock = nn.MultiheadAttention(self.embed_size, self.num_heads, \
                                              batch_first=True).to(self.config['DEVICE'])
        self.layernorm = LayerNormalisation(self.embed_size).to(self.config['DEVICE'])
        self.fclayers = [nn.Linear(self.num_heads*int(self.embed_size/self.num_heads),self.embed_size),\
                         nn.ReLU(),nn.Linear(self.embed_size,self.embed_size)]
        self.fc = nn.Sequential(*self.fclayers).to(self.config['DEVICE'])

    def forward(self,keys,query,mask=None):
        x,_ = self.mhablock(query,keys,keys)
        x += query
        x = self.layernorm(x)
        x0 = self.fc(x)
        x0 += x
        x0 = self.layernorm(x0)
        return x0

class Encoder(nn.Module):
    def __init__(self,config):
        super(Encoder,self).__init__()
        self.config = config
        self.n_layers = config['num_encoder_layers']
        self.patches = config['patches']
        self.pos_embed1D = nn.Parameter(torch.randn((1, self.patches, config['embed_size'])))
        self.layers = [EncoderLayer(config).to(self.config['DEVICE']) for _ in range(self.n_layers)]
        self.config = config

    def forward(self,input,mask=None):
        x = input
        pos_embed = self.pos_embed1D.repeat(input.shape[0],1,1)
        x += (pos_embed)
        for layer in self.layers:
            x = layer(x)
        return x

class Decoder(nn.Module):

    def __init__(self,config):
        super(Decoder, self).__init__()
        self.config = config
        self.n_layers = config['num_encoder_layers']
        self.patches = config['output_tokens']
        self.pos_embed1D = nn.Parameter(nn.Parameter(torch.randn(1, self.patches, config['embed_size'])))
        self.layers = DecoderLayer(self.config).to(self.config['DEVICE'])

    def forward(self,keys,query,mask=None,embedding = False):
        if embedding:
            pos_embed = self.pos_embed1D.repeat(query.shape[0], 1, 1)
            query += (pos_embed)
        x = self.layers(keys,query,mask=mask)
        return x

