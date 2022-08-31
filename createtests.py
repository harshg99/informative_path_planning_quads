# Creating 1000 tests maps to compare results

# Metrrics to compare against
# Rewards, coverage
import os

import numpy as np

from env.searchenv import *
from env.env_setter import *
from params import *
import getopt,sys
import Utilities
from pdb import set_trace as T
TESTS = 200

def create_test_reward_maps(env,nummaps:int,index=None,ID=None):
    if ID is not None:
        dir_name = os.getcwd() + '/' + MAP_TEST_DIR + '/' + args_dict['TEST_TYPE'].format(ID)+'/'
    else:
        dir_name = os.getcwd() + '/' + MAP_TEST_DIR + '/' + args_dict['TEST_TYPE'].format(30)+'/'
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
    if index is None:
        index = 0

    divergences = list()


    for j in range(nummaps):
        env.reset()
        file_name = dir_name+ "tests{}".format(j+index*nummaps)
        np.save(file_name+"env",env.rewardMap)
        if args_dict['ENV_TYPE']=='GPPrim':
            np.save(file_name+"target",env.targetMap)
            np.save(file_name+"target_orig_dist",env.orig_target_distribution_map)
            #T()
            # kl_divergence = np.mean(env.worldBeliefMap * np.log(
            #     np.clip(env.worldBeliefMap, 1e-10, 1) / np.clip(env.orig_target_distribution_map, 1e-10, 1)
            # ))
            kl_divergence = np.mean(np.square(env.worldBeliefMap-env.orig_target_distribution_map))
            divergences.append(kl_divergence)

    return divergences


if __name__=="__main__":
    map_size = 0
    ID = None
    try:
        opts,args = getopt.getopt(sys.argv[1:],"hn:",["--mapsize"])
    except getopt.GetoptError:
        print('runtests.py -n <map_size 0[30x30] 1[45x45] 2[60x60] 3[90x90]> ')
        sys.exit(2)

    for opt,arg in opts:
        if opt =='h':
            print('runtests.py -n <map_size 0[30x30] 1[45x45] 2[60x60] 3[90x90]>')
            sys.exit()
        elif opt in ("-n","-map size"):
            map_size = arg

    import params as args
    args_dict = Utilities.set_dict(parameters=args)
    environment = env_setter().set_env(args_dict)
    divergences = np.array(create_test_reward_maps(environment,TESTS,ID=60))
    divergences_mean = divergences.mean()
    divergences_std = divergences.std()
    print('Divergences Mean{} Std{} Max{} Min {}'.format(divergences_mean,
                                                         divergences_std,
                                                         divergences.max(),
                                                         divergences.min()))