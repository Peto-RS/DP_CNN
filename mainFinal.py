from final.Dataset import Dataset
from final.DatasetAnalyser import DatasetAnalyser
from final.GlobalSettings import GlobalSettings
from final.ModelTrain import ModelTrain
from final.ModelEvaluation import ModelEvaluation
from final.PyTorchModels import PyTorchModels
from enums.PyTorchModelsEnum import PyTorchModelsEnum

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

import os


def main():
    datasets_to_train = ['./data/hymenoptera_data']
    models_to_train = [PyTorchModelsEnum.RESNET18]

    for dataset_to_train in datasets_to_train:
        GlobalSettings.DATASET_DIR = dataset_to_train

        for model_name in models_to_train:
            GlobalSettings.CNN_MODEL = model_name
            GlobalSettings.TRAIN_DIR = GlobalSettings.DATASET_DIR + '/' + GlobalSettings.TRAIN_DIRNAME
            GlobalSettings.VALID_DIR = GlobalSettings.DATASET_DIR + '/' + GlobalSettings.VALID_DIRNAME
            GlobalSettings.TEST_DIR = GlobalSettings.DATASET_DIR + '/' + GlobalSettings.TEST_DIRNAME
            GlobalSettings.NUM_CLASSES = len(os.listdir(GlobalSettings.TRAIN_DIR))

            print('CNN MODEL NAME: ' + str(GlobalSettings.CNN_MODEL))

            model_ft = PyTorchModels.get_cnn_model(
                model_name=GlobalSettings.CNN_MODEL,
                num_classes=GlobalSettings.NUM_CLASSES,
                feature_extract=GlobalSettings.FEATURE_EXTRACT,
                use_pretrained=True
            )

            dataset = Dataset()
            DatasetAnalyser.show_dataset_analysis(save_on_disk=True)

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

            model_ft, history = ModelTrain.train_model(
                model=model_ft,
                dataloaders=dataset.data_loaders,
                criterion=criterion,
                optimizer=optimizer_ft,
                num_epochs=GlobalSettings.NUM_EPOCHS,
                is_inception=(GlobalSettings.CNN_MODEL == PyTorchModelsEnum.INCEPTION_V3)
            )

            ModelEvaluation.train_valid_graph(history=history, save_on_disk=True)

            criterion = nn.NLLLoss()
            results = ModelEvaluation.evaluate(model_ft, dataset.data_loaders[GlobalSettings.TEST_DIRNAME], criterion)
            # print(results.head())
            pd.DataFrame(results).to_csv(GlobalSettings.SAVE_FOLDER + "/" + GlobalSettings.CNN_MODEL.value[0] + "/stats.csv")

            cat_df = DatasetAnalyser.get_dataset_statistic()
            results = results.merge(cat_df, left_on='class', right_on='Category').drop(columns=['Category'])
            print(results.head())
            ModelEvaluation.get_top1_accuracy(results=results, save_on_disk=True)

            # Weighted column of test images
            results['weighted'] = results['n_test'] / results['n_test'].sum()

            # Create weighted accuracies
            for i in (1, GlobalSettings.NUM_CLASSES):
                results[f'weighted_top{i}'] = results['weighted'] * results[f'top{i}']

            # Find final accuracy accounting for frequencies
            top1_weighted = results['weighted_top1'].sum()
            loss_weighted = (results['weighted'] * results['loss']).sum()

            print(f'Final test cross entropy per image = {loss_weighted:.4f}.')
            print(f'Final test top 1 weighted accuracy = {top1_weighted:.2f}%')

            f = open('./' + GlobalSettings.SAVE_FOLDER + '/' + str(GlobalSettings.CNN_MODEL.value[0]) + '/testAcc.txt',
                     "w")
            f.write(str(top1_weighted) + '\n')
            f.close()


if __name__ == '__main__':
    main()
