from enum import Enum


class PyTorchModelsEnum(Enum):
    ALEXNET = 'alexnet',
    VGG11 = 'vgg11',
    VGG11_BN = "vgg11_bn",
    VGG13 = 'vgg13',
    VGG13_BN = 'vgg13_bn',
    VGG16 = 'vgg16',
    VGG16_BN = 'vgg16_bn',
    VGG19 = 'vgg19',
    VGG19_BN = 'vgg19_bn',
    RESNET18 = 'resnet18',
    RESNET34 = 'resnet34',
    RESNET50 = 'resnet50',
    RESNET101 = 'resnet101',
    RESNET152 = 'resnet152',
    SQUEEZENET1_0 = 'squeezenet1_0',
    SQUEEZENET1_1 = 'squeezenet1_1',
    DENSENET121 = 'densenet121',
    DENSENET161 = 'densenet161',
    DENSENET169 = 'densenet169',
    DENSENET201 = 'densenet201',
    INCEPTION_V3 = 'inception_v3',
    GOOGLENET = 'googlenet'
