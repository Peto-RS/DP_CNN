from PyQt5 import QtWidgets, uic
from PyQt5.QtCore import QObject, pyqtSignal

from classes.Dataset import Dataset
from classes.ModelTraining import ModelTraining


class TrainingThread(QObject):
    finished = pyqtSignal()

    console_append = pyqtSignal(str)
    console_clear = pyqtSignal()
    console_replace = pyqtSignal(str)

    label_epoch_current_total_text_changed = pyqtSignal(str)
    label_training_model_name_text_changed = pyqtSignal(str)
    plot_train_valid_acc_loss_graph = pyqtSignal(object, str)
    progressBar_training_set_value_changed = pyqtSignal(float)

    def __init__(self, app_model):
        super(TrainingThread, self).__init__()
        self.app_model = app_model

    def run(self):
        dataset, dataset_dataloader = Dataset.create_dataloader_for_neural_network(
            dataset_train_test_valid_directory=self.app_model.model['dataset_train_test_valid_directory'],
            dataset_train_dir_name=self.app_model.model['dataset_train_dir_name'],
            dataset_test_dir_name=self.app_model.model['dataset_test_dir_name'],
            dataset_is_test_set_enabled=self.app_model.model['dataset_is_test_set_enabled'],
            dataset_valid_dir_name=self.app_model.model['dataset_valid_dir_name'],
            dataset_data_augmentation_train_enabled=self.app_model.model['dataset_data_augmentation_train_enabled'],
            dataset_data_augmentation_train=self.app_model.model['dataset_data_augmentation_train'],
            dataset_data_augmentation_test_enabled=self.app_model.model['dataset_data_augmentation_test_enabled'],
            dataset_data_augmentation_test=self.app_model.model['dataset_data_augmentation_test'],
            dataset_data_augmentation_valid_enabled=self.app_model.model['dataset_data_augmentation_valid_enabled'],
            dataset_data_augmentation_valid=self.app_model.model['dataset_data_augmentation_valid'],
            training_batch_size=self.app_model.model['training_batch_size']
        )

        ModelTraining.train_models(dataset_dataloader=dataset_dataloader,
                                   dataset_test_dir_name=self.app_model.model.get(
                                       'dataset_test_dir_name'),
                                   dataset_train_dir_name=self.app_model.model.get(
                                       'dataset_train_dir_name'),
                                   dataset_valid_dir_name=self.app_model.model.get(
                                       'dataset_valid_dir_name'),
                                   training_cnn_models_to_train=self.app_model.model[
                                       'training_cnn_models_to_train'],
                                   training_criterion=self.app_model.model[
                                       'training_criterion'],
                                   training_dropout=self.app_model.model['training_dropout'],
                                   training_epochs_early_stopping=self.app_model.model[
                                       'training_epochs_early_stopping'],
                                   training_epochs_count=self.app_model.model[
                                       'training_epochs_count'],
                                   training_feature_extract=self.app_model.model[
                                       'training_feature_extract'],
                                   training_optimizer=self.app_model.model[
                                       'training_optimizer'],
                                   training_learning_rate=self.app_model.model[
                                       'training_learning_rate'],
                                   training_lr_gamma=self.app_model.model['training_lr_gamma'],
                                   training_lr_step_size=self.app_model.model[
                                       'training_lr_step_size'],
                                   training_model_output_directory=self.app_model.model[
                                       'training_model_output_directory'],
                                   training_momentum=self.app_model.model['training_momentum'],
                                   training_save_best_model_enabled=self.app_model.model[
                                       'training_save_best_model_enabled'],
                                   training_scheduler=self.app_model.model[
                                       'training_scheduler'],
                                   training_use_gpu=self.app_model.model['training_use_gpu'],
                                   training_use_early_stopping=self.app_model.model[
                                       'training_use_early_stopping'],
                                   training_use_pretrained_models=self.app_model.model[
                                       'training_use_pretrained_models'],
                                   training_use_softmax=self.app_model.model[
                                       'training_use_softmax'],
                                   signals={
                                       'console_append': self.console_append,
                                       'console_clear': self.console_clear,
                                       'console_replace': self.console_replace,
                                       'label_epoch_current_total_text_changed': self.label_epoch_current_total_text_changed,
                                       'label_training_model_name_text_changed': self.label_training_model_name_text_changed,
                                       'progressBar_training_set_value_changed': self.progressBar_training_set_value_changed,
                                       'plot_train_valid_acc_loss_graph': self.plot_train_valid_acc_loss_graph
                                   })

        self.finished.emit()
