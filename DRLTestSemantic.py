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
from env.searchenv import *
from copy import  deepcopy
import pandas as pd
from Utilities import set_dict
from env.GPSemantic import GPSemanticGym
from env.render import make_gif

class DRLTest:
    def __init__(self,args_dict: dict,model_path: str):

        self.args_dict = args_dict
        import env_params.Semantic as parameters
        env_params_dict = set_dict(parameters)
        env_params_dict['home_dir'] = "./"
        self.env_params_dict = env_params_dict


        self.gifs = self.args_dict['TEST_GIFS']

        self.gifs_path = self.args_dict['TEST_GIFS_PATH'].format(model_path)
        if not os.path.exists(self.gifs_path):
            os.makedirs(self.gifs_path)

        if self.gifs:
            self.args_dict['RENDER_TRAINING'] = True
            self.args_dict['RENDER_TRAINING_WINDOW'] = 1

        self.env = GPSemanticGym(env_params_dict, self.args_dict)
        self.model = alg_setter.set_model(self.env,self.args_dict)
        print('Loading Model')
        model_path = "data/models/" + model_path
        checkpoint = torch.load(model_path + "/checkpoint{}.pkl".format(args_dict['LOAD_BEST_MODEL']),map_location = self.args_dict['DEVICE'])
        self.model.load_state_dict(checkpoint['model_state_dict'])

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

    def run_test(self,test_map_ID, test_ID):
        '''
              Run a test episode
              @param test_map_ID: ID of the test map (0-24)
              @param test_ID: testing ID for this set of map (0-ENV_PARAMS['TEST_PER_MAP']-1)
              '''

        episode_step = 0.0
        episode_rewards = 0.0
        self.env.reset(episode_num=0, test_map=test_ID, test_indices=test_map_ID)
        observation = self.env.get_obs_all()
        frames = []
        done = False

        # kl_divergence = np.mean(self.env.worldBeliefMap*np.log(
        #     np.clip(self.env.worldBeliefMap,1e-10,1)/np.clip(orig_target_map_dist,1e-10,1)
        # ))
        # kl_divergence = np.mean(np.square(self.env.worldBeliefMap - orig_target_map_dist))
        hidden_in = None

        while ((not self.args_dict['FIXED_BUDGET'] and episode_step < self.env.episode_length) \
               or (self.args_dict['FIXED_BUDGET'])):


            if self.args_dict['LSTM']:
                action_dict,hidden_in = self.plan_action(observation,hidden_in)
            else:
                action_dict,_ = self.plan_action(observation,hidden_in)

            returns = self.env.step_all(action_dict)
            if len(returns) == 3:
                rewards, rewards_dict, done = returns
            else:
                rewards, done = returns
            episode_rewards += np.array(rewards).sum()
            observation = self.env.get_obs_all()

            if self.gifs:
                frames += self.env.render(mode='rgb_array')
            episode_step+=1
            if done:
                break

        metrics = self.env.get_final_metrics()
        metrics['episode_reward'] = episode_rewards
        metrics['episode_length'] = episode_step
        metrics['divergence'] = self.env.proximity
        if self.gifs:
            make_gif(np.array(frames),
                 '{}/episode_{:d}_{:d}_{:.1f}.gif'.format(self.gifs_path,
                                                          test_map_ID*self.env_params_dict['TEST_PER_MAP']+test_ID ,
                                                          0,
                                                          episode_rewards))
        return metrics


class Tests:
    def __init__(self):
        import params as args
        args_dict = Utilities.set_dict(args)
        #self.results_path = args_dict['TEST_RESULTS_PATH'].format(model_path)
        #self.results_path = None

    def unit_tests(self,testID:int,model_path:str):
        import params as args
        args_dict = Utilities.set_dict(args)
        args_dict['GPU'] = False
        args_dict['DEVICE'] = 'cpu'

        import env_params.Semantic as parameters
        env_params_dict = set_dict(parameters)
        env_params_dict['home_dir'] = "./"
        self.env_params_dict = env_params_dict

        results = []
        for j in range(self.env_params_dict['TEST_PER_MAP']):
            drl_planner = DRLTest(args_dict,model_path)
            self.model_path = model_path
            dir_name = os.getcwd() + "/" + MAP_TEST_DIR + '/' + TEST_TYPE.format(0) +'/'
            results.append(drl_planner.run_test(test_map_ID=testID,test_ID=j))



        return results

    def run_tests(self,num_tests:int,model_path:str,num_threads:int):
        import params as args
        args_dict = Utilities.set_dict(args)
        self.results_path = args_dict['TEST_RESULTS_PATH'].format(model_path, 0,args_dict['BUDGET'])
        with mp.Pool(num_threads) as pool:
            types = [model_path for _ in range(num_tests)]
            results = pool.starmap(self.unit_tests,zip(range(num_tests),types))
            return results

    def get_results_path(self,type:str):
        return self.results_path

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_tests', type=int, default=2)
    parser.add_argument('--type', type=str, default='Semantic')
    parser.add_argument('--num_threads', type=int, default=2)
    parser.add_argument('--model_path', type=str, default="ModelLSTM")

    return parser.parse_args()

if __name__=="__main__":
    # Defaulkt test params

    args = parse_args()

    TestObj  = Tests()
    results = TestObj.run_tests(args.num_tests,args.model_path,args.num_threads)
    results_cat = []

    for result in results:
        results_cat += result

    result_dict = {}
    for j,result in enumerate(results_cat):
        result_dict[j] = result

    #T()
    results_path = TestObj.get_results_path(args.model_path)+"/all_results.json"
    if not os.path.exists(TestObj.get_results_path(args.model_path)):
        os.makedirs(TestObj.get_results_path(args.model_path))
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
    results_path = TestObj.get_results_path(args.model_path)+"/compiled_results.json"
    out_file = open(results_path, "w")
    json.dump(compiled_results,out_file)
    out_file.close()

    data = pd.DataFrame(result_dict)
    results_path_csv = TestObj.get_results_path(args.model_path) + "/all_results.csv"
    data.to_csv(results_path_csv)