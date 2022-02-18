import numpy as np
import scipy.signal as signal
from params import *
import neptune.new as neptune

def get_sampled_actions(policy):
    action_dict = {}
    for k in range(policy.shape[1]):
        size = policy[0,k,:].shape[0]
        p = policy[0,k,:]
        a = np.random.choice(range(size), p=p.ravel())
        action_dict[k] = a
    return action_dict

def discount(x, gamma):
    return signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

def make_list_dict(metrics) :
    tb_dict = {}
    for k,v in metrics.items() :
        tb_dict[k] = {}
        for k2,v2 in metrics[k].items() :
            tb_dict[k][k2] = [v2]
    return tb_dict

def add_to_dict(tensorboardData,metrics):
    for k,v in metrics.items() :
        for k2,v2 in metrics[k].items() :
            tensorboardData[k][k2].append(v2)
    return tensorboardData

def get_mean_dict(tensorboardData) :
    for k,v in tensorboardData.items() :
        for k2,v2 in tensorboardData[k].items() :
            tensorboardData[k][k2] = np.nanmean(np.array(v2))
    return tensorboardData


class Tensorboard():
    def __init__(self, global_summary):
        self.window_size = SUMMARY_WINDOW
        self.last_update = 0
        self.tensorboardData = []
        self.global_summary = global_summary
        self.prev_metrics = None

    def update(self, metrics, currEpisode, run=None):
        if self.last_update == 0:
            self.tensorboardData = make_list_dict(metrics)
        else:
            self.tensorboardData = add_to_dict(self.tensorboardData, metrics)
        self.last_update += 1

        if self.last_update > self.window_size:
            self.writeToTensorBoard(currEpisode)
            self.writeToNeptune(run, currEpisode)
            self.last_update = 0
            self.tensorboardData = []

    def writeToTensorBoard(self, currEpisode):
        # each row in tensorboardData represents an episode
        # each column is a specific metric
        #TODO
        mean_data = get_mean_dict(self.tensorboardData)
        counter = 0
        for k, v in mean_data.items():
            for k2, v2 in mean_data[k].items():
                self.global_summary.add_scalar(tag='{}/{}'.format(k, k2), scalar_value=v2,global_step = currEpisode)

        return

    def writeToNeptune(self, run, currEpisode):
        if run is not None:
            mean_data = get_mean_dict(self.tensorboardData)
            for k, v in mean_data.items():
                for k2, v2 in mean_data[k].items():
                    run['training/{}/{}'.format(k, k2)].log(value=v2, step=currEpisode)
        return

def setup_neptune() :
    global run
    run = None
    if int(NEPTUNE) :
        if LOAD_MODEL :
            if NEPTUNE_RUN is not None :
                project = neptune_project
                token = NEPTUNE_API_TOKEN
                run = neptune.init(project=project, api_token=token,run=NEPTUNE_RUN )
                return run
            else :
                project = neptune_project
                token = NEPTUNE_API_TOKEN
                run = neptune.init(project=project, api_token=token, run=NEPTUNE_RUN)
                raise RuntimeError('Please specify run to resume from in Neptune')

        project = neptune_project
        token = NEPTUNE_API_TOKEN
        run = neptune.init(project=project, api_token=token, run=NEPTUNE_RUN)

    return run
