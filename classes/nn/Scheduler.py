import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

from enums.SchedulerEnum import SchedulerEnum

class Scheduler:
    @staticmethod
    def get_scheduler(optimizer, training_scheduler_name, training_lr_gamma, training_lr_step_size):
        if SchedulerEnum.STEP_LR == training_scheduler_name:
            return lr_scheduler.StepLR(optimizer, step_size=training_lr_step_size, gamma=training_lr_gamma)
