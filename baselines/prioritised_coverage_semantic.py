from env.searchenvMP import *
from env.env_setter import *
import numpy as np
from copy import deepcopy
from params import *
import os
from Utilities import set_dict
from cmaes import CMA
from multiprocessing import Pool as pool
import functools
from baselines.il_wrapper import il_wrapper
from baselines.coverage_planner_mp import coverage_planner_mp

class Node:
    def __init__(self,incoming,outgoing,state,map_state,current_cost,depth = None,cost_fn=None):
        self.incoming = incoming
        self.outgoing = outgoing
        self.state = state
        self.map_state = map_state
        self.cost_fn =  cost_fn
        self.current_cost = current_cost
        self.depth = None

    def __eq__(self, other):
        if self.incoming == other.incoming and self.outgoing == other.outgoing\
                and np.all(self.state==other.state) and self.depth==other.depth\
                and np.all(self.map_state==other.map_state):
            return True
        return False

    def __lt__(self,other):
        if self.heuristic_cost_fn(self.map_state) + self.current_cost <= \
                self.heuristic_cost_fn(other.map_state)+ other.current_cost:
            return True
        return False

    def heuristic_cost_fn(self):
        return self.cost_fn(self.map_state)


class prioritised_coverage_semantic(coverage_planner_mp):
    def __init__(self,params_dict,home_dir="/"):
        super().__init__(params_dict,home_dir)


    def getCoverage(self, visited_states,world_map):

        world_map_init = deepcopy(world_map)

        for state in visited_states.tolist():

            semantic_obs,_ = world_map.get_observations(state, fov=self.env.sensor_params['sensor_range'],
                                                       scale=None, type='semantic',
                                                       return_distance=False, resolution=None)

            projected_measurement = np.argmax(semantic_obs, axis=-1)

            world_map.update_semantics(state, projected_measurement, self.env.sensor_params)


        init_coverage = (world_map_init.coverage_map*world_map_init.semantic_map).mean(axis=-1).sum()
        final_coverage = (world_map.coverage_map*world_map.semantic_map).mean(axis=-1).sum()
        coverage = (final_coverage - init_coverage).sum()

        return coverage/(np.square(self.env.sensor_params['sensor_range'][0])*world_map.resolution**2)
        #return entropy_reduction



if __name__=="__main__":
    import baseline_params.PriorCoverageSemantic as parameters
    planner = prioritised_coverage_semantic(set_dict(parameters),home_dir='../')
    print(planner.run_test(test_map_ID=0,test_ID=0))
