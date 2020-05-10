import torch.nn as nn
from torchvision import models

from enums.PyTorchModelsEnum import PyTorchModelsEnum


class PyTorchModels:
    @staticmethod
    def set_parameter_requires_grad(model, feature_extracting):
        if feature_extracting:
            for param in model.parameters():
                param.requires_grad = False

    @staticmethod
    def get_classifier(num_ftrs, num_classes, training_dropout, training_use_softmax):
        if training_use_softmax:
            return nn.Sequential(nn.Dropout(training_dropout), nn.Linear(num_ftrs, num_classes), nn.LogSoftmax(dim=1))

        return nn.Sequential(nn.Dropout(training_dropout), nn.Linear(num_ftrs, num_classes))

    @staticmethod
    def get_cnn_model(model_id, use_pretrained, num_classes, feature_extract, training_dropout, training_use_softmax):
        model_ft = PyTorchModels.get_model(model_id=model_id, use_pretrained=use_pretrained)
        PyTorchModels.set_parameter_requires_grad(model_ft, feature_extract)

        if model_id == PyTorchModelsEnum.ALEXNET:
            num_ftrs = model_ft.classifier[6].in_features
            model_ft.classifier[6] = PyTorchModels.get_classifier(num_ftrs, num_classes, training_dropout,
                                                                  training_use_softmax)
        elif model_id == PyTorchModelsEnum.DENSENET121:
            num_ftrs = model_ft.classifier.in_features
            model_ft.classifier = PyTorchModels.get_classifier(num_ftrs, num_classes, training_dropout,
                                                               training_use_softmax)
        elif model_id == PyTorchModelsEnum.DENSENET161:
            num_ftrs = model_ft.classifier.in_features
            model_ft.classifier = PyTorchModels.get_classifier(num_ftrs, num_classes, training_dropout,
                                                               training_use_softmax)
        elif model_id == PyTorchModelsEnum.DENSENET169:
            num_ftrs = model_ft.classifier.in_features
            model_ft.classifier = PyTorchModels.get_classifier(num_ftrs, num_classes, training_dropout,
                                                               training_use_softmax)
        elif model_id == PyTorchModelsEnum.DENSENET201:
            num_ftrs = model_ft.classifier.in_features
            model_ft.classifier = PyTorchModels.get_classifier(num_ftrs, num_classes, training_dropout,
                                                               training_use_softmax)
        elif model_id == PyTorchModelsEnum.GOOGLENET:
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = PyTorchModels.get_classifier(num_ftrs, num_classes, training_dropout, training_use_softmax)

        elif model_id == PyTorchModelsEnum.INCEPTION_V3:
            num_ftrs = model_ft.AuxLogits.fc.in_features
            model_ft.AuxLogits.fc = PyTorchModels.get_classifier(num_ftrs, num_classes, training_dropout,
                                                                 training_use_softmax)
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = PyTorchModels.get_classifier(num_ftrs, num_classes, training_dropout, training_use_softmax)
        elif model_id == PyTorchModelsEnum.MOBILENET_V2:
            num_ftrs = model_ft.classifier[1].in_features
            model_ft.classifier[1] = PyTorchModels.get_classifier(num_ftrs, num_classes, training_dropout,
                                                                  training_use_softmax)
        elif model_id == PyTorchModelsEnum.MNASNET_0_5:
            num_ftrs = model_ft.classifier[1].in_features
            model_ft.classifier[1] = PyTorchModels.get_classifier(num_ftrs, num_classes, training_dropout,
                                                                  training_use_softmax)

        elif model_id == PyTorchModelsEnum.MNASNET_0_75:  # no pretrained
            num_ftrs = model_ft.classifier[1].in_features
            model_ft.classifier[1] = PyTorchModels.get_classifier(num_ftrs, num_classes, training_dropout,
                                                                  training_use_softmax)
        elif model_id == PyTorchModelsEnum.MNASNET_1_0:
            num_ftrs = model_ft.classifier[1].in_features
            model_ft.classifier[1] = PyTorchModels.get_classifier(num_ftrs, num_classes, training_dropout,
                                                                  training_use_softmax)
        elif model_id == PyTorchModelsEnum.MNASNET_1_3:
            num_ftrs = model_ft.classifier[1].in_features
            model_ft.classifier[1] = PyTorchModels.get_classifier(num_ftrs, num_classes, training_dropout,
                                                                  training_use_softmax)
        elif model_id == PyTorchModelsEnum.RESNET18:
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = PyTorchModels.get_classifier(num_ftrs, num_classes, training_dropout, training_use_softmax)
        elif model_id == PyTorchModelsEnum.RESNET34:
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = PyTorchModels.get_classifier(num_ftrs, num_classes, training_dropout, training_use_softmax)
        elif model_id == PyTorchModelsEnum.RESNET50:
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = PyTorchModels.get_classifier(num_ftrs, num_classes, training_dropout, training_use_softmax)
        elif model_id == PyTorchModelsEnum.RESNET101:
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = PyTorchModels.get_classifier(num_ftrs, num_classes, training_dropout, training_use_softmax)
        elif model_id == PyTorchModelsEnum.RESNET152:
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = PyTorchModels.get_classifier(num_ftrs, num_classes, training_dropout, training_use_softmax)
        elif model_id == PyTorchModelsEnum.RESNEXT50:
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = PyTorchModels.get_classifier(num_ftrs, num_classes, training_dropout, training_use_softmax)
        elif model_id == PyTorchModelsEnum.RESNEXT101:
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = PyTorchModels.get_classifier(num_ftrs, num_classes, training_dropout, training_use_softmax)
        elif model_id == PyTorchModelsEnum.SHUFFLENET_V2_0_5:
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = PyTorchModels.get_classifier(num_ftrs, num_classes, training_dropout, training_use_softmax)
        elif model_id == PyTorchModelsEnum.SHUFFLENET_V2_1_0:
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = PyTorchModels.get_classifier(num_ftrs, num_classes, training_dropout, training_use_softmax)
        elif model_id == PyTorchModelsEnum.SHUFFLENET_V2_1_5:
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = PyTorchModels.get_classifier(num_ftrs, num_classes, training_dropout, training_use_softmax)
        elif model_id == PyTorchModelsEnum.SHUFFLENET_V2_2_0:
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = PyTorchModels.get_classifier(num_ftrs, num_classes, training_dropout, training_use_softmax)
        elif model_id == PyTorchModelsEnum.SQUEEZENET1_0:
            model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
            model_ft.num_classes = num_classes
        elif model_id == PyTorchModelsEnum.SQUEEZENET1_1:
            model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
            model_ft.num_classes = num_classes
        elif model_id == PyTorchModelsEnum.VGG11:
            num_ftrs = model_ft.classifier[6].in_features
            model_ft.classifier[6] = PyTorchModels.get_classifier(num_ftrs, num_classes, training_dropout,
                                                                  training_use_softmax)
        elif model_id == PyTorchModelsEnum.VGG11_BN:
            num_ftrs = model_ft.classifier[6].in_features
            model_ft.classifier[6] = PyTorchModels.get_classifier(num_ftrs, num_classes, training_dropout,
                                                                  training_use_softmax)
        elif model_id == PyTorchModelsEnum.VGG13:
            num_ftrs = model_ft.classifier[6].in_features
            model_ft.classifier[6] = PyTorchModels.get_classifier(num_ftrs, num_classes, training_dropout,
                                                                  training_use_softmax)
        elif model_id == PyTorchModelsEnum.VGG13_BN:
            num_ftrs = model_ft.classifier[6].in_features
            model_ft.classifier[6] = PyTorchModels.get_classifier(num_ftrs, num_classes, training_dropout,
                                                                  training_use_softmax)
        elif model_id == PyTorchModelsEnum.VGG16:
            num_ftrs = model_ft.classifier[6].in_features
            model_ft.classifier[6] = PyTorchModels.get_classifier(num_ftrs, num_classes, training_dropout,
                                                                  training_use_softmax)
        elif model_id == PyTorchModelsEnum.VGG16_BN:
            num_ftrs = model_ft.classifier[6].in_features
            model_ft.classifier[6] = PyTorchModels.get_classifier(num_ftrs, num_classes, training_dropout,
                                                                  training_use_softmax)
        elif model_id == PyTorchModelsEnum.VGG19:
            num_ftrs = model_ft.classifier[6].in_features
            model_ft.classifier[6] = PyTorchModels.get_classifier(num_ftrs, num_classes, training_dropout,
                                                                  training_use_softmax)
        elif model_id == PyTorchModelsEnum.VGG19_BN:
            num_ftrs = model_ft.classifier[6].in_features
            model_ft.classifier[6] = PyTorchModels.get_classifier(num_ftrs, num_classes, training_dropout,
                                                                  training_use_softmax)
        elif model_id == PyTorchModelsEnum.WIDE_RESNET50:
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = PyTorchModels.get_classifier(num_ftrs, num_classes, training_dropout, training_use_softmax)
        elif model_id == PyTorchModelsEnum.WIDE_RESNET101:
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = PyTorchModels.get_classifier(num_ftrs, num_classes, training_dropout, training_use_softmax)

        return model_ft
                       
    @staticmethod
    def get_model(model_id, use_pretrained):
        model_ft = None
        if model_id == PyTorchModelsEnum.ALEXNET:
            model_ft = models.alexnet(pretrained=use_pretrained)
        elif model_id == PyTorchModelsEnum.DENSENET121:
            model_ft = models.densenet121(pretrained=use_pretrained)
        elif model_id == PyTorchModelsEnum.DENSENET161:
            model_ft = models.densenet161(pretrained=use_pretrained)
        elif model_id == PyTorchModelsEnum.DENSENET169:
            model_ft = models.densenet169(pretrained=use_pretrained)
        elif model_id == PyTorchModelsEnum.DENSENET201:
            model_ft = models.densenet201(pretrained=use_pretrained)
        elif model_id == PyTorchModelsEnum.GOOGLENET:
            model_ft = models.googlenet(pretrained=use_pretrained)
        elif model_id == PyTorchModelsEnum.INCEPTION_V3:
            model_ft = models.inception_v3(pretrained=use_pretrained)
        elif model_id == PyTorchModelsEnum.MOBILENET_V2:
            model_ft = models.mobilenet_v2(pretrained=use_pretrained)
        elif model_id == PyTorchModelsEnum.MNASNET_0_5:
            model_ft = models.mnasnet0_5(pretrained=use_pretrained)
        elif model_id == PyTorchModelsEnum.MNASNET_0_75:  # no pretrained
            model_ft = models.mnasnet0_75(pretrained=use_pretrained)
        elif model_id == PyTorchModelsEnum.MNASNET_1_0:
            model_ft = models.mnasnet1_0(pretrained=use_pretrained)
        elif model_id == PyTorchModelsEnum.MNASNET_1_3:
            model_ft = models.mnasnet1_3(pretrained=use_pretrained)
        elif model_id == PyTorchModelsEnum.RESNET18:
            model_ft = models.resnet18(pretrained=use_pretrained)
        elif model_id == PyTorchModelsEnum.RESNET34:
            model_ft = models.resnet34(pretrained=use_pretrained)
        elif model_id == PyTorchModelsEnum.RESNET50:
            model_ft = models.resnet50(pretrained=use_pretrained)
        elif model_id == PyTorchModelsEnum.RESNET101:
            model_ft = models.resnet101(pretrained=use_pretrained)
        elif model_id == PyTorchModelsEnum.RESNET152:
            model_ft = models.resnet152(pretrained=use_pretrained)
        elif model_id == PyTorchModelsEnum.RESNEXT50:
            model_ft = models.resnext50_32x4d(pretrained=use_pretrained)
        elif model_id == PyTorchModelsEnum.RESNEXT101:
            model_ft = models.resnext101_32x8d(pretrained=use_pretrained)
        elif model_id == PyTorchModelsEnum.SHUFFLENET_V2_0_5:
            model_ft = models.shufflenet_v2_x0_5(pretrained=use_pretrained)
        elif model_id == PyTorchModelsEnum.SHUFFLENET_V2_1_0:
            model_ft = models.shufflenet_v2_x1_0(pretrained=use_pretrained)
        elif model_id == PyTorchModelsEnum.SHUFFLENET_V2_1_5:
            model_ft = models.shufflenet_v2_x1_5(pretrained=use_pretrained)
        elif model_id == PyTorchModelsEnum.SHUFFLENET_V2_2_0:
            model_ft = models.shufflenet_v2_x2_0(pretrained=use_pretrained)
        elif model_id == PyTorchModelsEnum.SQUEEZENET1_0:
            model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        elif model_id == PyTorchModelsEnum.SQUEEZENET1_1:
            model_ft = models.squeezenet1_1(pretrained=use_pretrained)
        elif model_id == PyTorchModelsEnum.VGG11:
            model_ft = models.vgg11(pretrained=use_pretrained)
        elif model_id == PyTorchModelsEnum.VGG11_BN:
            model_ft = models.vgg11_bn(pretrained=use_pretrained)
        elif model_id == PyTorchModelsEnum.VGG13:
            model_ft = models.vgg13(pretrained=use_pretrained)
        elif model_id == PyTorchModelsEnum.VGG13_BN:
            model_ft = models.vgg13_bn(pretrained=use_pretrained)
        elif model_id == PyTorchModelsEnum.VGG16:
            model_ft = models.vgg16(pretrained=use_pretrained)
        elif model_id == PyTorchModelsEnum.VGG16_BN:
            model_ft = models.vgg16_bn(pretrained=use_pretrained)
        elif model_id == PyTorchModelsEnum.VGG19:
            model_ft = models.vgg19(pretrained=use_pretrained)
        elif model_id == PyTorchModelsEnum.VGG19_BN:
            model_ft = models.vgg19_bn(pretrained=use_pretrained)
        elif model_id == PyTorchModelsEnum.WIDE_RESNET50:
            model_ft = models.wide_resnet50_2(pretrained=use_pretrained)
        elif model_id == PyTorchModelsEnum.WIDE_RESNET101:
            model_ft = models.wide_resnet101_2(pretrained=use_pretrained)
        
        return model_ft
