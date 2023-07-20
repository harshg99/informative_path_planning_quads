#Network Params
obs_model = 'Conv' #Conv
if obs_model == 'MLP':
    hidden_sizes = [256,128,128,64]
elif obs_model == 'Conv':
    hidden_sizes = [16,32,64]
pos_layer_size = [8,16]
graph_node_layer_size = [32,64]
budget_layer_size = [4,16]
