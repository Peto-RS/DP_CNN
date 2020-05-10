import os
from datetime import datetime

import torch

from classes.PyTorchModels import PyTorchModels
from classes.Utils import Utils

DELIMITER = '_'


class ModelIO:
    @staticmethod
    def save(model, output_dir, model_id, classes, feature_extract, use_pretrained, training_dropout,
             training_use_softmax):
        output_dir = Utils.create_folder_if_not_exists(output_dir)

        filename = model_id + DELIMITER + datetime.now().strftime("%Y%m%d_%H%M%S") + '.pth'
        torch.save(model, os.path.join(output_dir, filename))
        torch.save({
            'classes': classes,
            'feature_extract': feature_extract,
            'model_id': model_id,
            'model_state_dict': model.state_dict(),
            'use_pretrained': use_pretrained,
            'training_dropout': training_dropout,
            'training_use_softmax': training_use_softmax
        }, os.path.join(output_dir, filename))

    @staticmethod
    def load(path):
        loaded_model = torch.load(path)
        model = PyTorchModels.get_cnn_model(model_id=loaded_model['model_id'],
                                            use_pretrained=False,
                                            num_classes=len(loaded_model['classes']),
                                            feature_extract=loaded_model['feature_extract'],
                                            training_dropout=loaded_model['training_dropout'],
                                            training_use_softmax=loaded_model['training_use_softmax'])
        model.load_state_dict(loaded_model['model_state_dict'])

        return loaded_model

    @staticmethod
    def save_checkpoint():
        print()

    @staticmethod
    def load_checkpoint():
        print()
