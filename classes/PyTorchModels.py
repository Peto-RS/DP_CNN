import torch.nn as nn
from torchvision import models

from enums.PyTorchModelsEnum import PyTorchModelsEnum


class PyTorchModels:
    def __init__(self):
        print()

    @staticmethod
    def set_parameter_requires_grad(model, feature_extracting):
        if feature_extracting:
            for param in model.parameters():
                param.requires_grad = False

    @staticmethod
    def get_cnn_model(model_name, num_classes, feature_extract, use_pretrained=True):
        model_ft = None
        input_size = 0

        if model_name == PyTorchModelsEnum.ALEXNET:
            model_ft = models.alexnet(pretrained=use_pretrained)
            PyTorchModels.set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.classifier[6].in_features
            model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
            input_size = 224

        elif model_name == PyTorchModelsEnum.RESNET18:
            model_ft = models.resnet18(pretrained=use_pretrained)
            PyTorchModels.set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, num_classes)
            input_size = 224

        elif model_name == PyTorchModelsEnum.RESNET34:
            model_ft = models.resnet34(pretrained=use_pretrained)
            PyTorchModels.set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, num_classes)
            input_size = 224

        elif model_name == PyTorchModelsEnum.RESNET50:
            model_ft = models.resnet50(pretrained=use_pretrained)
            PyTorchModels.set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, num_classes)
            input_size = 224

        elif model_name == PyTorchModelsEnum.RESNET101:
            model_ft = models.resnet101(pretrained=use_pretrained)
            PyTorchModels.set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, num_classes)
            input_size = 224

        elif model_name == PyTorchModelsEnum.RESNET152:
            model_ft = models.resnet152(pretrained=use_pretrained)
            PyTorchModels.set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, num_classes)
            input_size = 224

        elif model_name == PyTorchModelsEnum.VGG11:
            model_ft = models.vgg11(pretrained=use_pretrained)
            PyTorchModels.set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.classifier[6].in_features
            model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
            input_size = 224

        elif model_name == PyTorchModelsEnum.VGG11_BN:
            model_ft = models.vgg11_bn(pretrained=use_pretrained)
            PyTorchModels.set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.classifier[6].in_features
            model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
            input_size = 224

        elif model_name == PyTorchModelsEnum.VGG13:
            model_ft = models.vgg13(pretrained=use_pretrained)
            PyTorchModels.set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.classifier[6].in_features
            model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
            input_size = 224

        elif model_name == PyTorchModelsEnum.VGG13_BN:
            model_ft = models.vgg13_bn(pretrained=use_pretrained)
            PyTorchModels.set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.classifier[6].in_features
            model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
            input_size = 224

        elif model_name == PyTorchModelsEnum.VGG16:
            model_ft = models.vgg16(pretrained=use_pretrained)
            PyTorchModels.set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.classifier[6].in_features
            model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
            input_size = 224

        elif model_name == PyTorchModelsEnum.VGG16_BN:
            model_ft = models.vgg16_bn(pretrained=use_pretrained)
            PyTorchModels.set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.classifier[6].in_features
            model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
            input_size = 224

        elif model_name == PyTorchModelsEnum.VGG19:
            model_ft = models.vgg19(pretrained=use_pretrained)
            PyTorchModels.set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.classifier[6].in_features
            model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
            input_size = 224

        elif model_name == PyTorchModelsEnum.VGG19_BN:
            model_ft = models.vgg19_bn(pretrained=use_pretrained)
            PyTorchModels.set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.classifier[6].in_features
            model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
            input_size = 224

        elif model_name == PyTorchModelsEnum.SQUEEZENET1_0:
            model_ft = models.squeezenet1_0(pretrained=use_pretrained)
            PyTorchModels.set_parameter_requires_grad(model_ft, feature_extract)
            model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
            model_ft.num_classes = num_classes
            input_size = 224

        elif model_name == PyTorchModelsEnum.SQUEEZENET1_1:
            model_ft = models.squeezenet1_1(pretrained=use_pretrained)
            PyTorchModels.set_parameter_requires_grad(model_ft, feature_extract)
            model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
            model_ft.num_classes = num_classes
            input_size = 224

        elif model_name == PyTorchModelsEnum.DENSENET121:
            model_ft = models.densenet121(pretrained=use_pretrained)
            PyTorchModels.set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.classifier.in_features
            model_ft.classifier = nn.Linear(num_ftrs, num_classes)
            input_size = 224

        elif model_name == PyTorchModelsEnum.DENSENET161:
            model_ft = models.densenet161(pretrained=use_pretrained)
            PyTorchModels.set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.classifier.in_features
            model_ft.classifier = nn.Linear(num_ftrs, num_classes)
            input_size = 224

        elif model_name == PyTorchModelsEnum.DENSENET169:
            model_ft = models.densenet169(pretrained=use_pretrained)
            PyTorchModels.set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.classifier.in_features
            model_ft.classifier = nn.Linear(num_ftrs, num_classes)
            input_size = 224

        elif model_name == PyTorchModelsEnum.DENSENET201:
            model_ft = models.densenet201(pretrained=use_pretrained)
            PyTorchModels.set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.classifier.in_features
            model_ft.classifier = nn.Linear(num_ftrs, num_classes)
            input_size = 224

        elif model_name == PyTorchModelsEnum.INCEPTION_V3:
            model_ft = models.inception_v3(pretrained=use_pretrained)
            PyTorchModels.set_parameter_requires_grad(model_ft, feature_extract)

            num_ftrs = model_ft.AuxLogits.fc.in_features
            model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)

            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, num_classes)

            input_size = 299

        else:
            print("Unknown model name, exiting...")
            exit()

        return model_ft, input_size
