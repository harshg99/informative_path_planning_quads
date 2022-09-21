#Network Params
obs_model = 'MLP' #Conv
if obs_model == 'MLP':
    hidden_sizes1 = [4,8]
    hidden_sizes = [256,128,128,64]
elif obs_model == 'Conv':
    hidden_sizes = [16,32,64,128]
pos_layer_size = [8,16]


