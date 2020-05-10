from app_model.data_augmentation.DataAugmentationModelTest import DataAugmentationModelTest
from app_model.data_augmentation.DataAugmentationModelTrain import DataAugmentationModelTrain
from app_model.data_augmentation.DataAugmentationModelValid import DataAugmentationModelValid

from enums.CriterionEnum import CriterionEnum
from enums.OptimizerEnum import OptimizerEnum
from enums.PyTorchModelsEnum import PyTorchModelsEnum
from enums.SchedulerEnum import SchedulerEnum


class AppModel:
    _instance = None

    def __init__(self):
        self.model = dict()

        ###
        # dataset
        ###
        self.model['dataset_cleaning_enabled'] = True  # TODO
        self.model['dataset_data_augmentation_test'] = DataAugmentationModelTest()
        self.model['dataset_data_augmentation_test_enabled'] = True
        self.model['dataset_data_augmentation_train'] = DataAugmentationModelTrain()
        self.model['dataset_data_augmentation_train_enabled'] = True
        self.model['dataset_data_augmentation_valid'] = DataAugmentationModelValid()
        self.model['dataset_data_augmentation_valid_enabled'] = True
        self.model['dataset_diff_hash_threshold'] = 16  # TODO
        self.model['dataset_directory_with_classes'] = './saves/dataset_classes/minutiae'
        self.model['dataset_is_test_set_enabled'] = True
        self.model['dataset_test_dir_name'] = 'test'
        self.model['dataset_test_set_percentage'] = 15
        self.model['dataset_train_dir_name'] = 'train'
        self.model['dataset_train_set_percentage'] = 70
        self.model['dataset_train_test_valid_directory'] = './saves/dataset_train_test_valid/minutiae/minutiae_basic'
        self.model['dataset_valid_dir_name'] = 'valid'
        self.model['dataset_valid_set_percentage'] = 15

        ###
        # testing
        ###
        self.model['testing_dataset_class_name_selected'] = []
        self.model['testing_dataset_test_directory'] = './saves/dataset_train_test_valid/minutiae/test'
        self.model['testing_saved_models_directory'] = './saves/models'
        self.model['testing_saved_models_selected'] = []

        ###
        # training
        ###
        self.model['training_batch_size'] = 32
        self.model['training_epochs_count'] = 50
        self.model['training_cnn_models_to_train'] = [PyTorchModelsEnum.ALEXNET]
        self.model['training_criterion'] = CriterionEnum.CROSS_ENTROPY_LOSS
        self.model['training_dropout'] = 0.0
        self.model['training_epochs_early_stopping'] = 5
        self.model['training_evaluation_directory'] = './saves/evaluation'
        self.model['training_feature_extract'] = True
        self.model['training_learning_rate'] = 0.001
        self.model['training_lr_gamma'] = 0.1
        self.model['training_lr_step_size'] = 7
        self.model['training_model_output_directory'] = './saves/models'
        self.model['training_momentum'] = 0.9
        self.model['training_optimizer'] = OptimizerEnum.SGD
        self.model['training_save_best_model_enabled'] = True
        self.model['training_save_train_valid_graph'] = False
        self.model['training_scheduler'] = SchedulerEnum.STEP_LR
        self.model['training_use_early_stopping'] = True
        self.model['training_use_gpu'] = True
        self.model['training_use_pretrained_models'] = True
        self.model['training_use_softmax'] = False

    def set_to_model(self, key, value):
        self.model[key] = value

    @staticmethod
    def get_instance():
        if AppModel._instance is None:
            AppModel._instance = AppModel()
        return AppModel._instance
