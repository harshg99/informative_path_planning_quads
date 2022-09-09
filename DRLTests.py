import multiprocessing as mp
import numpy as np

from baselines import baseline_setter
import os
import sys,getopt
from params import *
import json
from pdb import set_trace as T
import Utilities
from models.alg_setter import alg_setter
from env.env_setter import env_setter
from copy import  deepcopy
import pandas as pd
class DRLTest:
    def __init__(self,args_dict: dict,model_path: str,map_size: int):

        self.args_dict = args_dict
        self.args_dict['GPU'] = False # CPU testing
        self.gifs = args_dict['TEST_GIFS']
        self.env = env_setter.set_env(args_dict)
        self.model = alg_setter.set_model(self.env, args_dict)
        self.gifs_path = self.args_dict['TEST_GIFS_PATH'].format(model_path,
                                                                 map_size,
                                                                 args_dict['BUDGET']
                                                                 )

        print('Loading Model')
        model_path = 'data/models/' + model_path
        checkpoint = torch.load(model_path + "/checkpoint.pkl",map_location = self.args_dict['DEVICE'])
        self.model.load_state_dict(checkpoint['model_state_dict'])
        #self.model.optim.load_state_dict(checkpoint['optimizer_state_dict'])
        curr_episode = checkpoint['epoch']
        print('Model results at Episode: {}'.format(curr_episode))


    def plan_action(self, observation,hidden_in=None):

        # print(observation)
        hidden_in = hidden_in
        if self.args_dict['LSTM']:
            policy, value,hidden_out = self.model.forward_step(observation,hidden_in)
            hidden_in = hidden_out.cpu().detach().numpy()
        else:
            policy, value = self.model.forward_step(observation)
        policy = policy.cpu().detach().numpy()
        action_dict = Utilities.best_actions(policy)

        return action_dict,hidden_in

    def run_test(self,rewardmap,ID=0,targetMap=None,orig_target_map_dist=None):
        episode_step = 0.0
        episode_rewards = 0.0
        np.random.seed(seed=ID)
        self.env.reset(rewardmap,targetMap,orig_target_map_dist)
        frames = []
        done = False
        observation = self.env.get_obs_all()
        # kl_divergence = np.mean(self.env.worldBeliefMap*np.log(
        #     np.clip(self.env.worldBeliefMap,1e-10,1)/np.clip(orig_target_map_dist,1e-10,1)
        # ))
        kl_divergence = np.mean(np.square(self.env.worldBeliefMap - orig_target_map_dist))
        hidden_in = None

        while ((not self.args_dict['FIXED_BUDGET'] and episode_step < self.env.episode_length) \
               or (self.args_dict['FIXED_BUDGET'])):
            if self.gifs:
                frames.append(self.env.render(mode='rgb_array'))

            if self.args_dict['LSTM']:
                action_dict,hidden_in = self.plan_action(observation,hidden_in)
            else:
                action_dict,_ = self.plan_action(observation,hidden_in)

            rewards,done = self.env.step_all(action_dict)
            episode_rewards += np.array(rewards).sum()
            observation = self.env.get_obs_all()

            episode_step+=1
            if done:
                break

        metrics = self.env.get_final_metrics()
        metrics['episode_reward'] = episode_rewards
        metrics['episode_length'] = episode_step
        metrics['divergence'] = kl_divergence
        if self.gifs:
            make_gif(np.array(frames),
                 '{}/episode_{:d}_{:d}_{:.1f}.gif'.format(self.gifs_path,ID , 0, episode_rewards))
        return metrics


class Tests:
    def __init__(self):
        import params as args
        args_dict = Utilities.set_dict(args)
        #self.results_path = args_dict['TEST_RESULTS_PATH'].format(model_path)
        self.results_path = None

    def unit_tests(self,testID:int,model_path:str,map_size:int):
        import params as args
        args_dict = Utilities.set_dict(args)
        args_dict['GPU'] = False
        args_dict['DEVICE'] = 'cpu'
        drl_planner = DRLTest(args_dict,model_path,map_size)
        self.model_path = model_path
        dir_name = os.getcwd() + "/" + MAP_TEST_DIR + '/' + TEST_TYPE.format(map_size) +'/'


        if ENV_TYPE=='GPPrim':
            file_name = dir_name + "tests{}env.npy".format(testID)
            rewardmap = np.load(file_name)
            file_name = dir_name + "tests{}target.npy".format(testID)
            targetmap = np.load(file_name)
            file_name =dir_name+"tests{}target_orig_dist.npy".format(testID)
            origtargetmap=np.load(file_name)
        else:
            file_name = dir_name + "tests{}.npy".format(testID)
            rewardmap = np.load(file_name)
            targetmap  = None

        return drl_planner.run_test(rewardmap,testID,targetmap,orig_target_map_dist=origtargetmap)

    def run_tests(self,type:str,num_tests:int,num_threads:int,mapSize:int):
        import params as args
        args_dict = Utilities.set_dict(args)
        self.results_path = args_dict['TEST_RESULTS_PATH'].format(model_path, map_size, args_dict['BUDGET'])
        with mp.Pool(num_threads) as pool:
            types = [type for _ in range(num_tests)]
            mapSize = [mapSize for _ in range(num_tests)]
            results = pool.starmap(self.unit_tests,zip(range(num_tests),types,mapSize))
            return results

    def get_results_path(self,type:str):
        return self.results_path



if __name__=="__main__":
    # Defaulkt test params
    num_tests  = 2
    num_threads = 2
    model_path = 'GreedyGP'
    map_size = 30

    try:
        opts,args = getopt.getopt(sys.argv[1:],"hn:p:t:s:",["--num_tests","--modelpath","--num_threads","--size"])
    except getopt.GetoptError:
        print('runtests.py -n <num_tests> -p <model_path in data/models> -t <num of threads> -s <size of map>')
        sys.exit(2)

    for opt,arg in opts:
        if opt =='h':
            print('runtests.py -n <num_tests> -p <model_path in data/models> -t <num of threads> -s <size of map>')
            sys.exit()
        elif opt in ("-n","--num_tests"):
            num_tests = int(arg)
        elif opt in ("-t","--num_threads"):
            num_threads = int(arg)
        elif opt in ("-p", "--model_path"):
            model_path = str(arg)
        elif opt in ("-s", "--size"):
            map_size = int(arg)

    TestObj  = Tests()
    results = TestObj.run_tests(model_path,num_tests,num_threads,map_size)
    result_dict = {}
    for j,result in enumerate(results):
        result_dict[j] = result

    #T()
    results_path = TestObj.get_results_path(model_path)+"/all_results.json"
    if not os.path.exists(TestObj.get_results_path(model_path)):
        os.makedirs(TestObj.get_results_path(model_path))
    out_file = open(results_path , "w")
    json.dump(result_dict,out_file)
    out_file.close()


    mean_results = {}
    max_results = {}
    std_results = {}
    min_results= {}
    all_results = {}
    len_results = len(list(result_dict.keys()))

    for id in result_dict.keys():
        for key in result_dict[id].keys():
            if key not in all_results.keys():
                all_results[key] = [result_dict[id][key]]
            else:
                all_results[key].append(result_dict[id][key])

    for key in all_results.keys():
        mean_results[key] = np.array(all_results[key]).mean(axis=-1)
        max_results[key] = np.array(all_results[key]).max(axis=-1)
        std_results[key] = np.array(all_results[key]).std(axis=-1)
        min_results[key] = np.array(all_results[key]).min(axis=-1)

        print("{} Mean : {} Std: {} Max: {} Min: {}".\
              format(key, mean_results[key],\
                     std_results[key],max_results[key],min_results[key]))

    compiled_results = {}
    compiled_results['mean'] = mean_results
    compiled_results['max'] = max_results
    compiled_results['min'] = min_results
    compiled_results['std'] = std_results
    results_path = TestObj.get_results_path(model_path)+"/compiled_results.json"
    out_file = open(results_path, "w")
    json.dump(compiled_results,out_file)
    out_file.close()

    data = pd.DataFrame(result_dict)
    results_path_csv = TestObj.get_results_path(model_path) + "/all_results.csv"
    data.to_csv(results_path_csv)