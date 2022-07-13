# Creating 1000 tests maps to compare results

# Metrrics to compare against
# Rewards, coverage
import os
from env.searchenv import *
from env.env_setter import *
from params import *
import getopt,sys
import Utilities

TESTS = 1000

def create_test_reward_maps(env,nummaps:int,index=None):
    dir_name = os.getcwd() + '/' + MAP_TEST_DIR + '/' + args_dict['ENV_TYPE']+'/'
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
    if index is None:
        index = 0
    for j in range(nummaps):
        env.reset()
        file_name = dir_name+ "tests{}".format(j+index*nummaps)
        np.save(file_name+"env",env.rewardMap)
        if args_dict['ENV_TYPE']=='GPPrim':
            np.save(file_name+"target",env.targetMap)
    pass


if __name__=="__main__":
    map_size = 0
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
    create_test_reward_maps(environment,TESTS)
