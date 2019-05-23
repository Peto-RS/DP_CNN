from final.Dataset import Dataset
from final.DatasetAnalyser import DatasetAnalyser
from final.GlobalSettings import GlobalSettings
from final.ModelTrain import ModelTrain
from final.ModelEvaluation import ModelEvaluation
from final.PyTorchModels import PyTorchModels
from enums.PyTorchModelsEnum import PyTorchModelsEnum

import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import os


def main():
    datasets_to_train = ['./data/caltech']
    models_to_train = [PyTorchModelsEnum.RESNET18]

    for dataset_to_train in datasets_to_train:
        GlobalSettings.DATASET_DIR = dataset_to_train

        for model_name in models_to_train:
            GlobalSettings.CNN_MODEL = model_name

            print('MODEL NAME: ' + str(GlobalSettings.CNN_MODEL))
            print('DATASET_DIR: ' + str(GlobalSettings.DATASET_DIR))

            model_ft = PyTorchModels.get_cnn_model(
                model_name=GlobalSettings.CNN_MODEL,
                num_classes=len(os.listdir(GlobalSettings.TRAIN_DIR)),
                feature_extract=GlobalSettings.FEATURE_EXTRACT,
                use_pretrained=True
            )

            dataset = Dataset()
            # DatasetAnalyser.show_dataset_analysis(save_on_disk=False)

            # Detect if we have a GPU available
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            model_ft = model_ft.to(device)

            params_to_update = model_ft.parameters()
            print("Params to learn:")
            if GlobalSettings.FEATURE_EXTRACT:
                params_to_update = []
                for name, param in model_ft.named_parameters():
                    if param.requires_grad == True:
                        params_to_update.append(param)
                        print("\t", name)
            else:
                for name, param in model_ft.named_parameters():
                    if param.requires_grad == True:
                        print("\t", name)

            optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)

            criterion = nn.CrossEntropyLoss()

            model_ft.class_to_idx = dataset.data['train'].class_to_idx
            model_ft.idx_to_class = {
                idx: class_
                for class_, idx in model_ft.class_to_idx.items()
            }

            model_ft, val_history = ModelTrain.train_model(
                model=model_ft,
                dataloaders=dataset.data_loaders,
                criterion=criterion,
                optimizer=optimizer_ft,
                num_epochs=GlobalSettings.NUM_EPOCHS,
                is_inception=(GlobalSettings.CNN_MODEL == PyTorchModelsEnum.INCEPTION_V3)
            )

            ohist = [h.cpu().numpy() for h in val_history]

            plt.title("Validation Accuracy vs. Number of Training Epochs")
            plt.xlabel("Training Epochs")
            plt.ylabel("Validation Accuracy")
            plt.plot(range(1, GlobalSettings.NUM_EPOCHS + 1), ohist, label="Pretrained")
            plt.ylim((0, 1.))
            plt.xticks(np.arange(1, GlobalSettings.NUM_EPOCHS + 1, 1.0))
            plt.legend()
            plt.show()

            criterion = nn.NLLLoss()
            results = ModelEvaluation.evaluate(model_ft, dataset.data_loaders[GlobalSettings.TEST_DIRNAME], criterion)
            print(results.head())


if __name__ == '__main__':
    main()
