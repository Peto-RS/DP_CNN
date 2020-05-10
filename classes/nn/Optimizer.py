import torch.nn as nn
import torch.optim as optim

from enums.OptimizerEnum import OptimizerEnum

class Optimizer:
    @staticmethod
    def get_optimizer(model, training_feature_extract, training_optimizer_name, training_learning_rate, training_momentum):
        params_to_update = Optimizer.get_params_to_update(model, training_feature_extract)

        if OptimizerEnum.SGD == training_optimizer_name:
            return optim.SGD(params_to_update, lr=training_learning_rate, momentum=training_momentum)
        elif OptimizerEnum.ADAM == training_optimizer_name:
            return optim.Adam(params_to_update, lr=training_learning_rate)

    @staticmethod
    def get_params_to_update(model, training_feature_extract):
        if training_feature_extract:
            params_to_update = []
            for name, param in model.named_parameters():
                if param.requires_grad:
                    params_to_update.append(param)

            return params_to_update

        return model.parameters()
