from enums.CriterionEnum import CriterionEnum
from enums.OptimizerEnum import OptimizerEnum
from enums.PyTorchModelsEnum import PyTorchModelsEnum
from enums.SchedulerEnum import SchedulerEnum


class GuiValues:
    _instance = None

    def __init__(self):
        self.model = dict()

        ###
        # dataset
        ###

        ###
        # training
        ###
        self.model['training_criterion_list'] = [
            CriterionEnum.CROSS_ENTROPY_LOSS,
            CriterionEnum.NLL_LOSS
        ]

        self.model['training_optimizer_list'] = [
            OptimizerEnum.ADAM,
            OptimizerEnum.SGD
        ]

        self.model['training_scheduler_list'] = [
            SchedulerEnum.STEP_LR
        ]

        self.model['train_cnn_list'] = [
            PyTorchModelsEnum.ALEXNET,
            PyTorchModelsEnum.DENSENET121,
            PyTorchModelsEnum.DENSENET161,
            PyTorchModelsEnum.DENSENET169,
            PyTorchModelsEnum.DENSENET201,
            PyTorchModelsEnum.GOOGLENET,
            PyTorchModelsEnum.INCEPTION_V3,
            PyTorchModelsEnum.MOBILENET_V2,
            PyTorchModelsEnum.MNASNET_0_5,
            PyTorchModelsEnum.MNASNET_0_75,
            PyTorchModelsEnum.MNASNET_1_0,
            PyTorchModelsEnum.MNASNET_1_3,
            PyTorchModelsEnum.RESNET18,
            PyTorchModelsEnum.RESNET34,
            PyTorchModelsEnum.RESNET50,
            PyTorchModelsEnum.RESNET101,
            PyTorchModelsEnum.RESNET152,
            PyTorchModelsEnum.RESNEXT50,
            PyTorchModelsEnum.RESNEXT101,
            PyTorchModelsEnum.SHUFFLENET_V2_0_5,
            PyTorchModelsEnum.SHUFFLENET_V2_1_0,
            PyTorchModelsEnum.SHUFFLENET_V2_1_5,
            PyTorchModelsEnum.SHUFFLENET_V2_2_0,
            PyTorchModelsEnum.SQUEEZENET1_0,
            PyTorchModelsEnum.SQUEEZENET1_1,
            PyTorchModelsEnum.VGG11,
            PyTorchModelsEnum.VGG11_BN,
            PyTorchModelsEnum.VGG13,
            PyTorchModelsEnum.VGG13_BN,
            PyTorchModelsEnum.VGG16,
            PyTorchModelsEnum.VGG16_BN,
            PyTorchModelsEnum.VGG19,
            PyTorchModelsEnum.VGG19_BN,
            PyTorchModelsEnum.WIDE_RESNET50,
            PyTorchModelsEnum.WIDE_RESNET101
        ]

    @staticmethod
    def get_instance():
        if GuiValues._instance is None:
            GuiValues._instance = GuiValues()
        return GuiValues._instance
