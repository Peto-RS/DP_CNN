from classes.CNN import CNN
from classes.InputDataAnalyser import InputDataAnalyser
from final.PyTorchModels import PyTorchModels

from enums.PyTorchModelsEnum import PyTorchModelsEnum

# import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms


def main():
    ##########
    # Input dataset params
    ##########
    data_dir = "data/hymenoptera_data"
    train_dir = data_dir + "/train/"
    valid_dir = data_dir + "/valid/"
    test_dir = data_dir + "/valid/"

    inputDatasetAnalyser = InputDataAnalyser(train_dir, valid_dir, test_dir)

    num_classes = len(os.listdir(train_dir))
    batch_size = 8
    num_epochs = 10
    feature_extract = False

    model_name = PyTorchModelsEnum.RESNET18

    model_ft, _ = PyTorchModels.get_cnn_model(model_name, num_classes, feature_extract, use_pretrained=True)

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # Create training and validation datasets
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'valid']}
    # Create training and validation dataloaders
    dataloaders_dict = {
        x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in
        ['train', 'valid']
    }

    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_ft = model_ft.to(device)

    params_to_update = model_ft.parameters()
    print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name, param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t", name)
    else:
        for name, param in model_ft.named_parameters():
            if param.requires_grad == True:
                print("\t", name)

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)


    # Setup the loss fxn
    criterion = nn.CrossEntropyLoss()

    # Train and evaluate
    model_ft, hist = CNN.train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs,
                                 is_inception=(model_name == "inception"))

    ohist = []
    shist = []

    ohist = [h.cpu().numpy() for h in hist]

    # plt.title("Validation Accuracy vs. Number of Training Epochs")
    # plt.xlabel("Training Epochs")
    # plt.ylabel("Validation Accuracy")
    # plt.plot(range(1, num_epochs + 1), ohist, label="Pretrained")
    # plt.ylim((0, 1.))
    # plt.xticks(np.arange(1, num_epochs + 1, 1.0))
    # plt.legend()
    # plt.show()


if __name__ == '__main__':
    main()
