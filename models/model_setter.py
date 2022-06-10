from models.Models import *
from models.DQN import *


class model_setter:

    @staticmethod
    def set_model(input_size,action_size,args_dict):
        type = args_dict['MODEL_TYPE']
        if type == 'Model1':
            import model_params.Model1 as parameters
            model_params_dict = model_setter.set_dict(parameters)
            return Model1(input_size,action_size,model_params_dict,args_dict)
        elif type == 'Model2':
            import model_params.Model2 as parameters
            model_params_dict = model_setter.set_dict(parameters)
            return Model2(input_size,action_size,model_params_dict,args_dict)
        elif type == 'Model3':
            import model_params.Model3 as parameters
            model_params_dict = model_setter.set_dict(parameters)
            return Model3(input_size, action_size, model_params_dict,args_dict)
        elif type == 'Model4':
            import model_params.Model4 as parameters
            model_params_dict = model_setter.set_dict(parameters)
            return Model4(input_size, action_size, model_params_dict,args_dict)

        elif type == 'Transformer1':
            from models.Transformer import TransformerAC
            import model_params.Transformer as parameters
            model_params_dict = model_setter.set_dict(parameters)
            return TransformerAC(input_size, action_size, model_params_dict,args_dict)
        elif type == 'Transformer2':
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