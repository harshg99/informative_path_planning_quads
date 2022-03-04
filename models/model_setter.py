from models.ActorCritic import *
from models.DQN import *

class model_setter:

    @staticmethod
    def set_model(input_size,action_size,type):
        if type == 'ActorCritic':
            import model_params.ActorCritic as parameters
            model_params_dict = model_setter.set_dict(parameters)
            return ActorCritic(input_size,action_size,model_params_dict)
        elif type == 'ActorCritic2':
            import model_params.ActorCritic as parameters
            model_params_dict = model_setter.set_dict(parameters)
            return ActorCritic2(input_size,action_size,model_params_dict)
        elif type == 'DQN':
            import model_params.ActorCritic as parameters
            model_params_dict = model_setter.set_dict(parameters)
            return DQN(input_size,action_size,model_params_dict)

    @staticmethod
    def set_dict(parameters):
        globals_dict = vars(parameters)
        new_dict = {}
        for k, v in globals_dict.items():
            if not k.startswith('__'):
                new_dict[k] = v
        return new_dict