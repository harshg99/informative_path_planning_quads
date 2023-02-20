import multiprocessing as mp
import numpy as np

from baselines import baseline_setter
import os
import sys,getopt
from params import *
import json
from pdb import set_trace as T
import pandas as pd
from Utilities import set_dict
class Tests:
    def unit_tests(self,type,test_map):
        baseline_planner = baseline_setter.baseline_setter.set_baseline_semantic(type)
        import env_params.Semantic as params
        params_dict = set_dict(params)

        results_list = []
        for j in range(params_dict['TEST_PER_MAP']):
            results_list.append(baseline_planner.run_test(test_map_ID=test_map, test_ID=j))

        return results_list

    def run_tests(self,num_tests:int,type:str,num_threads:int):
        with mp.Pool(num_threads) as pool:
            results = pool.starmap(self.unit_tests,zip([type for _ in range(num_tests)],range(num_tests)))
            return results

    def get_results_path(self,type:str):
        baseline_planner = baseline_setter.baseline_setter.set_baseline_semantic(type)
        return baseline_planner.results_path

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_tests', type=int, default=2)
    parser.add_argument('--type', type=str, default='Semantic')
    parser.add_argument('--num_threads', type=int, default=2)

    return parser.parse_args()

if __name__=="__main__":
    # Defaulkt test params

    args = parse_args()

    # Parsing command line arguments
    TestObj  = Tests()
    results = TestObj.run_tests(args.num_tests,args.type,args.num_threads)
    results_cat = []

    for result in results:
        results_cat += result

    result_dict = {}
    for j,result in enumerate(results_cat):
        result_dict[j] = results_cat

    #T()
    results_path = TestObj.get_results_path(args.type)+"/all_results.json"
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
    results_path = TestObj.get_results_path(args.type)+"/compiled_results.json"
    out_file = open(results_path, "w")
    json.dump(compiled_results,out_file)
    out_file.close()

    # outputting data to a csv file
    data = pd.DataFrame(result_dict)
    results_path_csv = TestObj.get_results_path(args.type) + "/all_results.csv"
    data.to_csv(results_path_csv)
