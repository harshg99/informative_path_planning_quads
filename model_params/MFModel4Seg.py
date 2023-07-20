obs_model = 'MLP' #Conv
# if obs_model == 'MLP':
#     hidden_sizes = [256,128,128,64]
# elif obs_model == 'Conv':
#     hidden_sizes = [16,32,64,128]
if obs_model=='MLP':
    hidden_sizes = [256,128,64,64]
else:
    hidden_sizes = [16,32,64]

pos_layer_size = [8,16]
graph_node_layer_size = [32,64]
budget_layer_size = [4,16]
#hidden_sizes         = [16,32]
pos_embed_bits     = 16
embed_size         = 64
num_heads            = 4
num_encoder_layers   = 4
token_length             = 16
gru_layers = 1