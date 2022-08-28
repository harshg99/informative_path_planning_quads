
class baseline_setter:

    @staticmethod
    def set_baseline(type):
        if type == 'Greedy':
            import baseline_params.GreedyMPparams as parameters
            from baselines.greedyMP import GreedyMP
            env_params_dict = baseline_setter.set_dict(parameters)
            return GreedyMP(env_params_dict)
        elif type == 'GreedyGP':
            import baseline_params.GreedyGPparams as parameters
            from baselines.greedyGP import GreedyGP
            env_params_dict = baseline_setter.set_dict(parameters)
            return GreedyGP(env_params_dict)
        elif type == 'CMAESGP':
            import baseline_params.CMAESGPparams as parameters
            from baselines.CMAESGP import CMAESGP
            env_params_dict = baseline_setter.set_dict(parameters)
            return CMAESGP(env_params_dict)
        elif type == 'CMAES':
            import baseline_params.CMAESparams as parameters
            from baselines.CMAES import CMAES
            env_params_dict = baseline_setter.set_dict(parameters)
            return CMAES(env_params_dict)
        elif type == 'coverage':
            import baseline_params.CoverageGPParams as parameters
            from baselines.coverage_planner_mp import coverage_planner_mp
            env_params_dict = baseline_setter.set_dict(parameters)
            return coverage_planner_mp(env_params_dict)
        elif type == 'prior_coverage':
            import baseline_params.CoverageGPParams as parameters
            from baselines.prioritised_coverage_mp import prioritised_coverage_mp
            env_params_dict = baseline_setter.set_dict(parameters)
            return prioritised_coverage_mp(env_params_dict)

    @staticmethod
    def set_dict(parameters):
        globals_dict = vars(parameters)
        new_dict = {}
        for k, v in globals_dict.items():
            if not k.startswith('__'):
                new_dict[k] = v
        return new_dict