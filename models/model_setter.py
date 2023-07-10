from models.Models import *
from models.DQN import *

class model_setter:

    @staticmethod
    def set_model(env,args_dict):
        type = args_dict['MODEL_TYPE']
        if type == 'Model1':
            import model_params.Model1 as parameters
            model_params_dict = model_setter.set_dict(parameters)
            return Model1(env,model_params_dict,args_dict)
        elif type == 'Model2':
            import model_params.Model2 as parameters
            model_params_dict = model_setter.set_dict(parameters)
            return Model2(env,model_params_dict,args_dict)
        elif type == 'Model3':
            import model_params.Model3 as parameters
            model_params_dict = model_setter.set_dict(parameters)
            return Model3(env, model_params_dict,args_dict)
        elif type == 'Model4':
            import model_params.Model4 as parameters
            model_params_dict = model_setter.set_dict(parameters)
            return Model4(env, model_params_dict,args_dict)
        elif type == 'Model5':
            import model_params.Model5 as parameters
            model_params_dict = model_setter.set_dict(parameters)
            return Model5(env, model_params_dict,args_dict)
        elif type == 'Model6':
            import model_params.Model5 as parameters
            model_params_dict = model_setter.set_dict(parameters)
            return Model6(env, model_params_dict,args_dict)
        elif type == 'Model6Seg':
            import model_params.Model5 as parameters
            model_params_dict = model_setter.set_dict(parameters)
            return Model6Seg(env, model_params_dict,args_dict)
        elif type == 'Transformer1':
            from models.Transformer import TransformerAC
            import model_params.Transformer as parameters
            model_params_dict = model_setter.set_dict(parameters)
            return TransformerAC(env, model_params_dict,args_dict)
        elif type == 'Transformer2':
            from models.Transformer import TransformerAC2
            import model_params.Transformer as parameters
            model_params_dict = model_setter.set_dict(parameters)
            return TransformerAC2(env, model_params_dict,args_dict)
        elif type == 'ModelTrans1':
            from models.ModelsTrans2 import ModelTrans1
            import model_params.ModelTrans1 as parameters
            model_params_dict = model_setter.set_dict(parameters)
            return ModelTrans1(env, model_params_dict,args_dict)
        elif type == 'ModelTrans2':
            from models.ModelsTrans2 import ModelTrans2
            import model_params.ModelTrans2 as parameters
            model_params_dict = model_setter.set_dict(parameters)
            return ModelTrans2(env, model_params_dict,args_dict)
        elif type == 'ModelMF1':
            import model_params.MFModel  as parameters
            from models.MFModels import ModelMF1
            model_params_dict = model_setter.set_dict(parameters)
            return ModelMF1(env,model_params_dict,args_dict)
        elif type == 'ModelMF2':
            import model_params.MFModel2  as parameters
            from models.MFModels import ModelMF2
            model_params_dict = model_setter.set_dict(parameters)
            return ModelMF2(env,model_params_dict,args_dict)
        elif type == 'ModelMF3':
            import model_params.MFModel2  as parameters
            from models.MFModels import ModelMF3
            model_params_dict = model_setter.set_dict(parameters)
            return ModelMF3(env,model_params_dict,args_dict)
        elif type == 'ModelMF4':
            import model_params.MFModel4  as parameters
            from models.MFModels import ModelMF4
            model_params_dict = model_setter.set_dict(parameters)
            return ModelMF4(env,model_params_dict,args_dict)
        elif type == 'ModelMF4Seg':
            import model_params.MFModel4Seg  as parameters
            from models.MFModels import ModelMF4Seg
            model_params_dict = model_setter.set_dict(parameters)
            return ModelMF4Seg(env,model_params_dict,args_dict)
        elif type == 'DQN':
            import model_params.Transformer as parameters
            model_params_dict = model_setter.set_dict(parameters)
            return DQN(env,model_params_dict,args_dict)

    @staticmethod
    def set_dict(parameters):
        globals_dict = vars(parameters)
        new_dict = {}
        for k, v in globals_dict.items():
            if not k.startswith('__'):
                new_dict[k] = v
        return new_dict