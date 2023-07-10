import numpy as np
import scipy.signal as signal
from params import *
import neptune.new as neptune

def sample_actions(policy):
    actions= {}
    for k in range(policy.shape[1]):
        size = policy[0,k,:].shape[0]
        p = policy[0,k,:]
        a = np.random.choice(range(size), p=p.ravel())
        actions[k] = a
    return actions

def best_actions(policy):
    actions= {}
    for k in range(policy.shape[1]):
        p = policy[0,k,:]
        a = np.argmax(p.ravel())
        actions[k] = a
    return actions

def discount(x, gamma):
    return signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

def make_list_dict(metrics) :
    dict = {}
    for k,v in metrics.items() :
        dict[k] = {}
        for k2,v2 in metrics[k].items() :
            dict[k][k2] = [v2]
    return dict

def append_dict(data,metrics):
    for k,v in metrics.items() :
        for k2,v2 in metrics[k].items() :
            data[k][k2].append(v2)
    return data

def get_means(data) :
    for k,v in data.items() :
        for k2,v2 in data[k].items() :
            data[k][k2] = np.nanmean(np.array(v2))
    return data


class Tensorboard():
    def __init__(self, global_summary):
        self.window_size = SUMMARY_WINDOW
        self.last_update = 0
        self.data = []
        self.global_summary = global_summary
        self.prev_metrics = None

    def update(self, metrics, currEpisode, run=None):
        if self.last_update == 0:
            self.data = make_list_dict(metrics)
        else:
            self.data = append_dict(self.data, metrics)
        self.last_update += 1

        if self.last_update > self.window_size:
            self.writeToTensorBoard(currEpisode)
            self.writeToNeptune(run, currEpisode)
            self.last_update = 0
            self.data = []

    def writeToTensorBoard(self, currEpisode):
        # each row in tensorboardData represents an episode
        # each column is a specific metric
        mean_data = get_means(self.data)
        counter = 0
        for k, v in mean_data.items():
            for k2, v2 in mean_data[k].items():
                self.global_summary.add_scalar(tag='{}/{}'.format(k, k2), scalar_value=v2,global_step = currEpisode)
        return

    def writeToNeptune(self, run, currEpisode):
        if run is not None:
            mean_data = get_means(self.data)
            for k, v in mean_data.items():
                for k2, v2 in mean_data[k].items():
                    run['training/{}/{}'.format(k, k2)].log(value=v2, step=currEpisode)
        return

def setup_neptune(args) :
    global run
    run = None
    if int(NEPTUNE) :
        if LOAD_MODEL  and NEPTUNE_RUN is None :
            if NEPTUNE_RUN is not None :
                raise RuntimeError('Please specify run to resume from in Neptune')

        project = args['neptune_project']
        token = args['NEPTUNE_API_TOKEN']
        run = neptune.init(project=project, api_token=token, run=NEPTUNE_RUN)
        run['params'] = args

    return run


def lambda_return(rewards,values,gamma,lamb):
    '''

    :param rewards: Rewards (batch,sequence)
    :param values: Values (batch,sequence)
    :param gamma: Discount
    :param lamb: Lamb weight
    :return: Lambda returns (batch,sequence)
    '''

    #shape (batch,T,T)
    rewards = rewards.copy()
    lambret = np.zeros(rewards.shape[-1])
    for j in range(rewards.shape[-1]):
        index = rewards.shape[-1] - j - 1


        if j ==0:
            if len(rewards.shape)==2:
                lambret[:,index] = rewards[:,index] + gamma*values[:,index+1]
            else:
                lambret[index] = rewards[index] + gamma * values[index + 1]
        else:
            if len(rewards.shape) == 2:
                lambret[:,index] = rewards[:,index] + gamma*(1-lamb)*values[:,index+1]\
                                + lamb*gamma*lambret[:,index+1]
            else:
                lambret[index] = rewards[index] + gamma * (1 - lamb) * values[index + 1] \
                                + lamb * gamma * lambret[index + 1]
    return  lambret

def set_dict(parameters):
    globals_dict = vars(parameters)
    new_dict = {}
    for k, v in globals_dict.items():
        if not k.startswith('__'):
            new_dict[k] = v
    return new_dict