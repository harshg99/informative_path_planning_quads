from models.ActorCritic import *
from models.DQN import *
import params as arguments

class model_setter:

    @staticmethod
    def set_model(input_size,action_size,type):
        args_dict = model_setter.set_dict(arguments)
        if type == 'ActorCritic':
            import model_params.ActorCritic as parameters
            model_params_dict = model_setter.set_dict(parameters)
            return ActorCritic(input_size,action_size,model_params_dict,args_dict)
        elif type == 'ActorCritic2':
            import model_params.ActorCritic as parameters
            model_params_dict = model_setter.set_dict(parameters)
            return ActorCritic2(input_size,action_size,model_params_dict,args_dict)
        elif type == 'ActorCritic3':
            import model_params.ActorCritic3 as parameters
            model_params_dict = model_setter.set_dict(parameters)
            return ActorCritic3(input_size, action_size, model_params_dict,args_dict)
        elif type == 'ActorCritic4':
            import model_params.ActorCritic4 as parameters
            model_params_dict = model_setter.set_dict(parameters)
            return ActorCritic4(input_size, action_size, model_params_dict,args_dict)
        elif type == 'ActorCritic5':
            import model_params.ActorCritic5 as parameters
            model_params_dict = model_setter.set_dict(parameters)
            return ActorCritic5(input_size, action_size, model_params_dict,args_dict)
        elif type == 'ActorCritic6':
            import model_params.ActorCritic6 as parameters
            model_params_dict = model_setter.set_dict(parameters)
            return ActorCritic6(input_size, action_size, model_params_dict,args_dict)
        elif type == 'TransformerAC':
            from models.Transformer import TransformerAC
            import model_params.Transformer as parameters
            model_params_dict = model_setter.set_dict(parameters)
            return TransformerAC(input_size, action_size, model_params_dict,args_dict)
        elif type == 'TransformerAC2':
            from models.Transformer import TransformerAC2
            import model_params.Transformer as parameters
            model_params_dict = model_setter.set_dict(parameters)
            return TransformerAC2(input_size, action_size, model_params_dict,args_dict)
        elif type == 'DQN':
            import model_params.Transformer as parameters
            model_params_dict = model_setter.set_dict(parameters)
            return DQN(input_size,action_size,model_params_dict,args_dict)

    @staticmethod
    def set_dict(parameters):
        globals_dict = vars(parameters)
        new_dict = {}
        for k, v in globals_dict.items():
            if not k.startswith('__'):
                new_dict[k] = v
        return new_dict