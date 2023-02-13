import numpy as np
from copy import deepcopy
'''
Maintians a list of performance metrics
'''

class Metrics:
    def __init__(self,initWorldMap=None,initTargetMap=None):
        self.initWorldMap = initWorldMap
        self.initTargetMap = initTargetMap

        self.coverage = None
        self.targetsfound = None
        self.map_entropy = None


    def update(self,initWorldMap,initTargetMap):
        assert initTargetMap is not None and initWorldMap is not None
        self.initWorldMap = initWorldMap.copy()
        self.initTargetMap = initTargetMap.copy()
        self.initial_entropy =  self._map_entropy(self.initWorldMap).sum()
        self.num_targets = np.sum(self.initTargetMap == 1)

    def compute_coverage_metric(self,beliefMap,targetMap):
        covered_cells = np.abs((beliefMap-self.initWorldMap))>0.001
        coverage = np.sum(covered_cells)/self.initWorldMap.shape[0]/self.initWorldMap.shape[1]*100
        return coverage

    def compute_targets_found(self,beliefMap,targetMap):
        num_target_found = np.sum(targetMap==2)
        targetsfound = num_target_found/self.num_targets*100
        return targetsfound

    def compute_map_entropy(self,beliefMap,targetMap):
        current_entropy = self._map_entropy(beliefMap).sum()
        return (1 - current_entropy/self.initial_entropy)*100

    def compute_metrics(self,beliefMap,targetMap):
        self.coverage = self.compute_coverage_metric(beliefMap,targetMap)
        self.map_entropy = self.compute_map_entropy(beliefMap,targetMap)
        self.targetsfound = self.compute_targets_found(beliefMap, targetMap)
        metrics = dict()
        metrics['coverage'] = self.coverage
        metrics['map_entropy_reduction'] = self.map_entropy
        metrics['targets_found'] = self.targetsfound
        return metrics

    def _map_entropy(self,beliefMap):
        return beliefMap*np.log(np.clip(beliefMap,1e-7,1.0)) +\
               (1-beliefMap)*np.log(np.clip(1-beliefMap,1e-7,1.0))


class SemanticMetrics:
    def __init__(self):
        pass

    def update(self, init_belief_map):
        assert init_belief_map is not None
        self.init_belief_map =  deepcopy(init_belief_map)
        self.initial_entropy = init_belief_map.get_entropy().sum()

    def compute_coverage_metric(self, final_map, ground_truth_map):
        return (final_map.coverage_map.sum() - self.init_belief_map.coverage_map.sum())\
               /(np.prod(self.init_belief_map.coverage_map.shape))*100

    def compute_semantics_found(self, belief_map, target_map):
        semantic_change_proportion = {}
        total_semantic_det = 0
        total_semantics_gt = 0
        for sem in target_map.semantic_list:
            semantic_change_proportion[sem] = np.sum(target_map.detected_semantic_map
                                                     [belief_map.detected_semantic_map == sem]) \
                                              / target_map.semantic_proportion[sem]

            total_semantic_det += np.sum(target_map.detected_semantic_map
                                                     [belief_map.detected_semantic_map == sem])
            total_semantics_gt += target_map.semantic_proportion[sem]

        return total_semantic_det/total_semantics_gt*100,semantic_change_proportion

    def compute_map_entropy(self, belief_map, target_map):
        current_entropy = belief_map.get_entropy().sum()
        return (1 - current_entropy / self.initial_entropy) * 100

    def compute_metrics(self, belief_map, target_map):
        '''
        beliefMap: current belief map
        targetMap: current fround truth target map
        '''
        self.coverage = self.compute_coverage_metric(belief_map, target_map)
        self.map_entropy = self.compute_map_entropy(belief_map, target_map)
        self.targetsfound,_ = self.compute_semantics_found(belief_map, target_map)
        metrics = dict()
        metrics['coverage'] = self.coverage
        metrics['map_entropy_reduction'] = self.map_entropy
        metrics['semantics_found'] = self.targetsfound
        return metrics