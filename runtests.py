import multiprocessing as mp
import numpy as np

from baselines import baseline_setter
import os
import sys,getopt
from params import *
import json
from pdb import set_trace as T

class Tests:
    def unit_tests(self,type:str,testID:int,mapSize:int):
        baseline_planner = baseline_setter.baseline_setter.set_baseline(type)
        dir_name = os.getcwd() + "/" + MAP_TEST_DIR + '/' + TEST_TYPE.format(mapSize) +'/'
        if ENV_TYPE=='GPPrim':
            file_name = dir_name + "tests{}env.npy".format(testID)
            rewardmap = np.load(file_name)
            file_name = dir_name + "tests{}target.npy".format(testID)
            targetmap = np.load(file_name)
            file_name = dir_name + "tests{}target_orig_dist.npy".format(testID)
            orig_target_map = np.load(file_name)
        else:
            file_name = dir_name + "tests{}.npy".format(testID)
            rewardmap = np.load(file_name)
            targetmap  = None
            orig_target_map = None

        return baseline_planner.run_test(rewardmap,testID,targetmap,orig_target_map)

    def run_tests(self,type:str,num_tests:int,num_threads:int,mapSize:int):
        with mp.Pool(num_threads) as pool:
            types = [type for _ in range(num_tests)]
            mapSize = [mapSize for _ in range(num_tests)]
            results = pool.starmap(self.unit_tests,zip(types,range(num_tests),mapSize))
            return results

    def get_results_path(self,type:str):
        baseline_planner = baseline_setter.baseline_setter.set_baseline(type)
        return baseline_planner.results_path



if __name__=="__main__":
    # Defaulkt test params
    num_tests  = 2
    num_threads = 2
    type = 'GreedyGP'
    map_size = 30

    try:
        opts,args = getopt.getopt(sys.argv[1:],"hn:t:p:s:",["--num_tests","--type","--num_threads","--size"])
    except getopt.GetoptError:
        print('runtests.py -n <num_tests> -t <type of test(Greedy,CMAES> -p <num of threads>  -s <size of map>')
        sys.exit(2)

    for opt,arg in opts:
        if opt =='h':
            print('runtests.py -n <num_tests> -t <type of test(Greedy,CMAES> -p <num of threads> -s <size of map>')
            sys.exit()
        elif opt in ("-n","--num_tests"):
            num_tests = int(arg)
        elif opt in ("-t","--type"):
            type = str(arg)
        elif opt in ("-p", "--num_threads"):
            num_threads = int(arg)
        elif opt in ("-s", "--size"):
            map_size = int(arg)

    TestObj  = Tests()
    results = TestObj.run_tests(type,num_tests,num_threads,map_size)
    result_dict = {}
    for j,result in enumerate(results):
        result_dict[j] = result

    #T()
    results_path = TestObj.get_results_path(type)+"/all_results.json"
    out_file = open(results_path, "w")
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
    results_path = TestObj.get_results_path(type)+"/compiled_results.json"
    out_file = open(results_path, "w")
    json.dump(compiled_results,out_file)
    out_file.close()