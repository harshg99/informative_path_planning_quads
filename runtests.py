import multiprocessing as mp
import numpy as np

from baselines import baseline_setter
import os
import sys,getopt
from params import *
import json

class Tests:
    def unit_tests(self,type:str,testID:int):
        baseline_planner = baseline_setter.baseline_setter.set_baseline(type)
        dir_name = os.getcwd() + "/" + MAP_TEST_DIR
        file_name = dir_name + "tests{}.npy".format(testID)
        rewardmap = np.load(file_name)
        return baseline_planner.run_test(rewardmap,testID)

    def run_tests(self,type:str,num_tests:int,num_threads:int):
        with mp.Pool(num_threads) as pool:
            types = [type for _ in range(num_tests)]
            results = pool.starmap(self.unit_tests,zip(types,range(num_tests)))
            return results



if __name__=="__main__":
    # Defaulkt test params
    num_tests  = 100
    num_threads = 15
    type = 'Greedy'

    try:
        opts,args = getopt.getopt(sys.argv[1:],"hn:t:p",["--num_tests","--type","--num_threads"])
    except getopt.GetoptError:
        print('runtests.py -n <num_tests> -t <type of test(Greedy,CMAES> -p <num of threads>')
        sys.exit(2)

    for opt,arg in opts:
        if opt =='h':
            print('runtests.py -n <num_tests> -t <type of test(Greedy,CMAES> -p <num of threads>')
            sys.exit()
        elif opt in ("-n","--num_tests"):
            num_tests = arg
        elif opt in ("-t","--type"):
            type = arg
        elif opt in ("-p", "--num_threads"):
            num_threads = arg

    TestObj  = Tests()
    results = TestObj.run_tests(type,num_tests,num_threads)
    results = np.array(results)

    print("Mean : {} Std: {} Max: {} Min: {}".\
          format(results.mean(axis=-1),\
                 results.std(axis=-1),results.max(axis=-1),results.min(axis=-1)))


