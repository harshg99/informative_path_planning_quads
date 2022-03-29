
from env.searchenv import *
from env.searchenvMP import *
class env_setter:

    @staticmethod
    def set_env(type):
        if type == 'Discrete':
            import env_params.Discrete as parameters
            env_params_dict = env_setter.set_dict(parameters)
            return SearchEnv(env_params_dict)
        elif type == 'MotionPrim':
            import env_params.MotionPrim as parameters
            env_params_dict = env_setter.set_dict(parameters)
            return SearchEnvMP(env_params_dict)


    @staticmethod
    def set_dict(parameters):
        globals_dict = vars(parameters)
        new_dict = {}
        for k, v in globals_dict.items():
            if not k.startswith('__'):
                new_dict[k] = v
        return new_dict