import numpy as np
'''
Maintians a list of performance metrics
'''

class Metrics:
    def __init__(self,initWorldMap=None,initTargetMap=None):
        self.initWorldMap = initWorldMap
        self.initTargetMap = initTargetMap
        self.num_targets = np.sum(self.initTargetMap==1)
        self.coverage = None
        self.targetsfound = None
        self.map_entropy = None


    def update(self,initWorldMap,initTargetMap):
        assert initTargetMap is not None and initWorldMap is not None
        self.initWorldMap = initWorldMap.copy()
        self.initTargetMap = initTargetMap.copy()
        self.initial_entropy =  self._map_entropy(self.initWorldMap)

    def compute_coverage_metric(self,beliefMap,targetMap):
        covered_cells = (beliefMap-self.initWorldMap)>0.001
        coverage = np.sum(covered_cells)/self.initWorldMap.shape[0]/self.initWorldMap.shape[1]*100
        return coverage

    def compute_targets_found(self,beliefMap,targetMap):
        num_target_found = np.sum(targetMap==2)
        targetsfound = num_target_found/self.num_targets*100
        return targetsfound

    def compute_map_entropy(self,beliefMap,targetMap):
        current_entropy = self._map_entropy(beliefMap)
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
