# Creating 1000 tests maps to compare results

# Metrrics to compare against
# Rewards, coverage
import os
from env.searchenv import *
from env.env_setter import *
from params import *


ENVTYPE = 'Discrete'
TESTS = 1000
def create_test_reward_maps(env,nummaps:int):
    dir_name = os.getcwd() + '/' + MAP_TEST_DIR
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
    for j in range(nummaps):
        env.reset()
        file_name = dir_name+ "tests{}".format(j)
        np.save(file_name,env.rewardMap)
    pass

if __name__=="__main__":
    environment = env_setter().set_env(ENVTYPE)
    create_test_reward_maps(environment,TESTS)