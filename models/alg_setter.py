from models.OnPolicy import *
from models.OffPolicy import *

class alg_setter:

    @staticmethod
    def set_model(env,args_dict):
        if args_dict['ALG_TYPE'] == 'AC':
            import model_params.AC as parameters
            model_params_dict = alg_setter.set_dict(parameters)
            return AC(env,model_params_dict,args_dict)
        elif args_dict['ALG_TYPE']  == 'PPO':
            import model_params.PPO as parameters
            model_params_dict = alg_setter.set_dict(parameters)
            return PPO(env,model_params_dict, args_dict)
        elif args_dict['ALG_TYPE'] == 'SAC':
            import model_params.SAC as parameters
            model_params_dict = alg_setter.set_dict(parameters)
            return SAC(env, model_params_dict, args_dict)
        if args_dict['ALG_TYPE'] == 'ACLSTM':
            import model_params.AC as parameters
            model_params_dict = alg_setter.set_dict(parameters)
            return ACLSTM(env,model_params_dict,args_dict)
        elif args_dict['ALG_TYPE']  == 'PPOLSTM':
            import model_params.PPO as parameters
            model_params_dict = alg_setter.set_dict(parameters)
            return PPOLSTM(env,model_params_dict, args_dict)
        elif args_dict['ALG_TYPE'] == 'SACLSTM':
            import model_params.SAC as parameters
            model_params_dict = alg_setter.set_dict(parameters)
            return SACLSTM(env, model_params_dict, args_dict)


    @staticmethod
    def set_dict(parameters):
        globals_dict = vars(parameters)
        new_dict = {}
        for k, v in globals_dict.items():
            if not k.startswith('__'):
                new_dict[k] = v
        return new_dict