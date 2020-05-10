import torch.nn as nn

from enums.CriterionEnum import CriterionEnum


class Criterion:
    @staticmethod
    def get_criterion(training_criterion_name):
        if CriterionEnum.CROSS_ENTROPY_LOSS == training_criterion_name:
            return nn.CrossEntropyLoss()
        elif CriterionEnum.NLL_LOSS == training_criterion_name:
            return nn.NLLLoss()
