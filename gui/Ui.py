import os

import torch
from torch.utils.data.dataloader import DataLoader

import webbrowser

from classes.Dataset import Dataset
from classes.DatasetAnalyser import DatasetAnalyser
from classes.ModelEvaluation import ModelEvaluation
from classes.ModelIO import ModelIO
from classes.PyTorchModels import PyTorchModels
from classes.TrainTestValidDatasetCreator import TrainTestValidDatasetCreator
from classes.Utils import Utils

from gui.DialogDataAugmentation import DialogDataAugmentation
from gui.DialogTraining import DialogTraining

from app_model.AppModel import AppModel
from app_model.GuiValues import GuiValues

from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import QFileDialog, QListWidget
from PyQt5.QtCore import Qt, QThread

from utils.PrettyPrintConfusionMatrix import PrettyPrintConfusionMatrix


class Ui(QtWidgets.QMainWindow):
    def __init__(self):
        super(Ui, self).__init__()
        uic.loadUi('gui/MainWindow.ui', self)

        self.dialog_training = None

        self.app_model = AppModel.get_instance()
        self.gui_values = GuiValues.get_instance()

        self.checkbox_dataset_is_test_set_enabled = self.findChild(QtWidgets.QCheckBox,
                                                                   'checkbox_dataset_is_test_set_enabled')
        self.checkBox_dataset_data_augmentation_train_enabled = self.findChild(QtWidgets.QCheckBox,
                                                                               'checkBox_dataset_data_augmentation_train_enabled')
        self.checkBox_dataset_data_augmentation_test_enabled = self.findChild(QtWidgets.QCheckBox,
                                                                              'checkBox_dataset_data_augmentation_test_enabled')
        self.checkBox_dataset_data_augmentation_valid_enabled = self.findChild(QtWidgets.QCheckBox,
                                                                               'checkBox_dataset_data_augmentation_valid_enabled')
        self.checkBox_training_feature_extract = self.findChild(QtWidgets.QCheckBox,
                                                                'checkBox_training_feature_extract')
        self.checkBox_training_save_best_model_enabled = self.findChild(QtWidgets.QCheckBox,
                                                                        'checkBox_training_save_best_model_enabled')
        self.checkBox_training_save_train_valid_graph = self.findChild(QtWidgets.QCheckBox,
                                                                       'checkBox_training_save_train_valid_graph')
        self.checkBox_training_use_early_stopping = self.findChild(QtWidgets.QCheckBox,
                                                                   'checkBox_training_use_early_stopping')
        self.checkBox_training_use_pretrained_models = self.findChild(QtWidgets.QCheckBox,
                                                                      'checkBox_training_use_pretrained_models')
        self.checkBox_training_use_softmax = self.findChild(QtWidgets.QCheckBox,
                                                            'checkBox_training_use_softmax')
        self.checkBox_training_use_gpu = self.findChild(QtWidgets.QCheckBox, 'checkBox_training_use_gpu')
        self.comboBox_training_criterion = self.findChild(QtWidgets.QComboBox, 'comboBox_training_criterion')
        self.comboBox_training_optimizer = self.findChild(QtWidgets.QComboBox, 'comboBox_training_optimizer')
        self.comboBox_training_scheduler = self.findChild(QtWidgets.QComboBox, 'comboBox_training_scheduler')
        self.doubleSpinBox_training_dropout = self.findChild(QtWidgets.QDoubleSpinBox,
                                                             'doubleSpinBox_training_dropout')
        self.doubleSpinBox_training_learning_rate = self.findChild(QtWidgets.QDoubleSpinBox,
                                                                   'doubleSpinBox_training_learning_rate')
        self.doubleSpinBox_training_lr_gamma = self.findChild(QtWidgets.QDoubleSpinBox,
                                                              'doubleSpinBox_training_lr_gamma')
        self.doubleSpinBox_training_lr_step_size = self.findChild(QtWidgets.QDoubleSpinBox,
                                                                  'doubleSpinBox_training_lr_step_size')
        self.doubleSpinBox_training_momentum = self.findChild(QtWidgets.QDoubleSpinBox,
                                                              'doubleSpinBox_training_momentum')
        self.lineedit_dataset_directory_with_classes = self.findChild(QtWidgets.QLineEdit,
                                                                      'lineedit_dataset_directory_with_classes')
        self.lineedit_dataset_train_test_valid_directory = self.findChild(QtWidgets.QLineEdit,
                                                                          'lineedit_dataset_train_test_valid_directory')
        self.lineedit_training_evaluation_directory = self.findChild(QtWidgets.QLineEdit,
                                                                     'lineedit_training_evaluation_directory')
        self.lineedit_training_model_output_directory = self.findChild(QtWidgets.QLineEdit,
                                                                       'lineedit_training_model_output_directory')
        self.lineedit_testing_dataset_test_directory = self.findChild(QtWidgets.QLineEdit,
                                                                      'lineedit_testing_dataset_test_directory')
        self.lineedit_testing_saved_models_directory = self.findChild(QtWidgets.QLineEdit,
                                                                      'lineedit_testing_saved_models_directory')
        self.listWidget_training_cnn_models_to_train = self.findChild(QtWidgets.QListWidget,
                                                                      'listWidget_training_cnn_models_to_train')
        self.listWidget_testing_dataset_class_name = self.findChild(QtWidgets.QListWidget,
                                                                    'listWidget_testing_dataset_class_name')
        self.listWidget_testing_saved_models = self.findChild(QtWidgets.QListWidget,
                                                              'listWidget_testing_saved_models')
        self.pushButton_run_samples_distribution = self.findChild(QtWidgets.QPushButton,
                                                                  'pushButton_run_samples_distribution')
        self.pushButton_training = self.findChild(QtWidgets.QPushButton, 'pushButton_training')
        self.pushButton_data_augmentation_train = self.findChild(QtWidgets.QPushButton,
                                                                 'pushButton_data_augmentation_train')
        self.pushButton_data_augmentation_test = self.findChild(QtWidgets.QPushButton,
                                                                'pushButton_data_augmentation_test')
        self.pushButton_data_augmentation_valid = self.findChild(QtWidgets.QPushButton,
                                                                 'pushButton_data_augmentation_valid')
        self.pushButton_dataset_show_analysis = self.findChild(QtWidgets.QPushButton,
                                                                 'pushButton_dataset_show_analysis')
        self.pushButton_testing_run = self.findChild(QtWidgets.QPushButton,
                                                     'pushButton_testing_run')
        self.pushButton_testing_confusion_matrix = self.findChild(QtWidgets.QPushButton,
                                                                  'pushButton_testing_confusion_matrix')
        self.pushButton_testing_confusion_matrix_class = self.findChild(QtWidgets.QPushButton,
                                                                        'pushButton_testing_confusion_matrix_class')
        self.pushButton_testing_roc_curve = self.findChild(QtWidgets.QPushButton,
                                                                        'pushButton_testing_roc_curve')
        self.pushButton_testing_top_k_accuracy = self.findChild(QtWidgets.QPushButton,
                                                                        'pushButton_testing_top_k_accuracy')
        self.spinbox_dataset_train_set_percentage = self.findChild(QtWidgets.QSpinBox,
                                                                   'spinbox_dataset_train_set_percentage')
        self.spinbox_dataset_test_set_percentage = self.findChild(QtWidgets.QSpinBox,
                                                                  'spinbox_dataset_test_set_percentage')
        self.spinbox_dataset_valid_set_percentage = self.findChild(QtWidgets.QSpinBox,
                                                                   'spinbox_dataset_valid_set_percentage')
        self.spinBox_training_batch_size = self.findChild(QtWidgets.QSpinBox, 'spinBox_training_batch_size')
        self.spinBox_training_epochs_count = self.findChild(QtWidgets.QSpinBox, 'spinBox_training_epochs_count')
        self.spinBox_training_epochs_early_stopping = self.findChild(QtWidgets.QSpinBox,
                                                                     'spinBox_training_epochs_early_stopping')
        self.toolbutton_dataset_directory_with_classes = self.findChild(QtWidgets.QToolButton,
                                                                        'toolbutton_dataset_directory_with_classes')
        self.toolbutton_dataset_train_test_valid_directory = self.findChild(QtWidgets.QToolButton,
                                                                            'toolbutton_dataset_train_test_valid_directory')
        self.toolbutton_testing_dataset_test_directory = self.findChild(QtWidgets.QToolButton,
                                                                        'toolbutton_testing_dataset_test_directory')
        self.toolbutton_testing_saved_models_directory = self.findChild(QtWidgets.QToolButton,
                                                                        'toolbutton_testing_saved_models_directory')
        self.toolbutton_training_evaluation_directory = self.findChild(QtWidgets.QToolButton,
                                                                       'toolbutton_training_evaluation_directory')
        self.toolbutton_training_model_output_directory = self.findChild(QtWidgets.QToolButton,
                                                                         'toolbutton_training_model_output_directory')
        self.init_gui()

        self.show()

    def init_gui(self):
        self.set_checkbox_dataset_is_test_set_enabled()
        self.set_checkBox_dataset_data_augmentation_train_enabled()
        self.set_checkBox_dataset_data_augmentation_test_enabled()
        self.set_checkBox_dataset_data_augmentation_valid_enabled()
        self.set_checkBox_training_feature_extract()
        self.set_checkBox_training_save_best_model_enabled()
        self.set_checkBox_training_save_train_valid_graph()
        self.set_checkBox_training_use_early_stopping()
        self.set_checkBox_training_use_pretrained_models()
        self.set_checkBox_training_use_gpu()
        self.set_checkBox_training_use_softmax()
        self.set_comboBox_training_criterion()
        self.set_comboBox_training_optimizer()
        self.set_comboBox_training_scheduler()
        self.set_doubleSpinBox_training_dropout()
        self.set_doubleSpinBox_training_learning_rate()
        self.set_doubleSpinBox_training_lr_gamma()
        self.set_doubleSpinBox_training_lr_step_size()
        self.set_doubleSpinBox_training_momentum()
        self.set_lineedit_dataset_directory_with_classes()
        self.set_lineedit_dataset_train_test_valid_directory()
        self.set_lineedit_testing_dataset_test_directory()
        self.set_lineedit_testing_saved_models_directory()
        self.set_lineedit_training_evaluation_directory()
        self.set_lineedit_training_model_output_directory()
        self.set_listWidget_training_cnn_models_to_train()
        self.set_listWidget_testing_dataset_class_name()
        self.set_listWidget_testing_saved_models()
        self.set_pushButton_dataset_show_analysis()
        self.set_pushButton_run_samples_distribution()
        self.set_pushButton_training()
        self.set_pushButton_data_augmentation_train()
        self.set_pushButton_data_augmentation_test()
        self.set_pushButton_data_augmentation_valid()
        self.set_pushButton_testing_run()
        self.set_pushButton_testing_confusion_matrix()
        self.set_pushButton_testing_confusion_matrix_class()
        self.set_pushButton_testing_roc_curve()
        self.set_pushButton_testing_top_k_accuracy()
        self.set_spinbox_dataset_train_set_percentage()
        self.set_spinbox_dataset_test_set_percentage()
        self.set_spinbox_dataset_valid_set_percentage()
        self.set_spinBox_training_batch_size()
        self.set_spinBox_training_epochs_early_stopping()
        self.set_spinBox_training_epochs_count()
        self.set_toolbutton_dataset_directory_with_classes()
        self.set_toolbutton_dataset_train_test_valid_directory()
        self.set_toolbutton_testing_dataset_test_directory()
        self.set_toolbutton_testing_saved_models_directory()
        self.set_toolbutton_training_model_output_directory()
        self.set_toolbutton_training_evaluation_directory()

    ###
    # checkbox
    ###
    def set_checkbox_dataset_is_test_set_enabled(self):
        self.checkbox_dataset_is_test_set_enabled.setChecked(self.app_model.model['dataset_is_test_set_enabled'])
        self.checkbox_dataset_is_test_set_enabled.stateChanged.connect(
            self.checkbox_dataset_is_test_set_enabled_state_changed)

    def checkbox_dataset_is_test_set_enabled_state_changed(self):
        new_value = self.checkbox_dataset_is_test_set_enabled.isChecked()
        self.app_model.set_to_model('dataset_is_test_set_enabled', new_value)
        if self.spinbox_dataset_valid_set_percentage:
            self.spinbox_dataset_valid_set_percentage.setReadOnly(not new_value)

    def set_checkBox_dataset_data_augmentation_train_enabled(self):
        self.checkBox_dataset_data_augmentation_train_enabled.setChecked(
            self.app_model.model['dataset_data_augmentation_train_enabled'])
        self.checkBox_dataset_data_augmentation_train_enabled.stateChanged.connect(
            self.checkBox_dataset_data_augmentation_train_enabled_state_changed)

    def checkBox_dataset_data_augmentation_train_enabled_state_changed(self):
        new_value = self.checkBox_dataset_data_augmentation_train_enabled.isChecked()
        self.app_model.set_to_model('dataset_data_augmentation_train_enabled', new_value)

    def set_checkBox_dataset_data_augmentation_test_enabled(self):
        self.checkBox_dataset_data_augmentation_test_enabled.setChecked(
            self.app_model.model['dataset_data_augmentation_test_enabled'])
        self.checkBox_dataset_data_augmentation_test_enabled.stateChanged.connect(
            self.checkBox_dataset_data_augmentation_test_enabled_state_changed)

    def checkBox_dataset_data_augmentation_test_enabled_state_changed(self):
        new_value = self.checkBox_dataset_data_augmentation_train_enabled.isChecked()
        self.app_model.set_to_model('dataset_data_augmentation_test_enabled', new_value)

    def set_checkBox_dataset_data_augmentation_valid_enabled(self):
        self.checkBox_dataset_data_augmentation_valid_enabled.setChecked(
            self.app_model.model['dataset_data_augmentation_valid_enabled'])
        self.checkBox_dataset_data_augmentation_valid_enabled.stateChanged.connect(
            self.checkBox_dataset_data_augmentation_valid_enabled_state_changed)

    def checkBox_dataset_data_augmentation_valid_enabled_state_changed(self):
        new_value = self.checkBox_dataset_data_augmentation_valid_enabled.isChecked()
        self.app_model.set_to_model('dataset_data_augmentation_valid_enabled', new_value)

    def set_checkBox_training_feature_extract(self):
        self.checkBox_training_feature_extract.setChecked(self.app_model.model['training_feature_extract'])
        self.checkBox_training_feature_extract.stateChanged.connect(
            self.checkBox_training_feature_extract_state_changed)

    def checkBox_training_feature_extract_state_changed(self):
        new_value = self.checkBox_training_feature_extract.isChecked()
        self.app_model.set_to_model('training_feature_extract', new_value)

    def set_checkBox_training_save_best_model_enabled(self):
        self.checkBox_training_save_best_model_enabled.setChecked(
            self.app_model.model['training_save_best_model_enabled'])
        self.checkBox_training_save_best_model_enabled.stateChanged.connect(
            self.set_checkBox_training_save_best_model_enabled_state_changed)

    def set_checkBox_training_save_best_model_enabled_state_changed(self):
        new_value = self.checkBox_training_save_best_model_enabled.isChecked()
        self.app_model.set_to_model('training_save_best_model_enabled', new_value)

    def set_checkBox_training_save_train_valid_graph(self):
        self.checkBox_training_save_train_valid_graph.setChecked(
            self.app_model.model['training_save_train_valid_graph'])
        self.checkBox_training_save_train_valid_graph.stateChanged.connect(
            self.checkBox_training_save_train_valid_graph_state_changed)

    def checkBox_training_save_train_valid_graph_state_changed(self):
        new_value = self.checkBox_training_save_train_valid_graph.isChecked()
        self.app_model.set_to_model('training_save_train_valid_graph', new_value)

    def set_checkBox_training_use_early_stopping(self):
        self.checkBox_training_use_early_stopping.setChecked(
            self.app_model.model['training_use_early_stopping'])
        self.checkBox_training_use_early_stopping.stateChanged.connect(
            self.checkBox_training_use_early_stopping_state_changed)

    def checkBox_training_use_early_stopping_state_changed(self):
        new_value = self.checkBox_training_use_early_stopping.isChecked()
        self.app_model.set_to_model('training_use_early_stopping', new_value)

    def set_checkBox_training_use_pretrained_models(self):
        self.checkBox_training_use_pretrained_models.setChecked(self.app_model.model['training_use_pretrained_models'])
        self.checkBox_training_use_pretrained_models.stateChanged.connect(
            self.checkBox_training_use_pretrained_models_state_changed)

    def checkBox_training_use_pretrained_models_state_changed(self):
        new_value = self.checkBox_training_use_pretrained_models.isChecked()
        self.app_model.set_to_model('training_use_pretrained_models', new_value)

    def set_checkBox_training_use_gpu(self):
        self.checkBox_training_use_gpu.setChecked(self.app_model.model['training_use_gpu'])
        self.checkBox_training_use_gpu.stateChanged.connect(self.checkBox_training_use_gpu_state_changed)

    def checkBox_training_use_gpu_state_changed(self):
        new_value = self.checkBox_training_use_gpu.isChecked()
        new_value = self.checkBox_training_use_gpu.isChecked()
        self.app_model.set_to_model('training_use_gpu', new_value)

    def set_checkBox_training_use_softmax(self):
        self.checkBox_training_use_softmax.setChecked(self.app_model.model['training_use_softmax'])
        self.checkBox_training_use_softmax.stateChanged.connect(self.checkBox_training_use_softmax_state_changed)

    def checkBox_training_use_softmax_state_changed(self):
        new_value = self.checkBox_training_use_softmax.isChecked()
        self.app_model.set_to_model('training_use_softmax', new_value)

    ###
    # combobox
    ###
    def set_comboBox_training_criterion(self):
        self.comboBox_training_criterion.addItems(self.gui_values.model['training_criterion_list'])

        index = self.comboBox_training_criterion.findText(self.app_model.model['training_criterion'],
                                                          Qt.MatchFixedString)
        if index >= 0:
            self.comboBox_training_criterion.setCurrentIndex(index)

        self.comboBox_training_criterion.activated.connect(self.comboBox_training_criterion_activated)

    def comboBox_training_criterion_activated(self):
        self.app_model.set_to_model('training_criterion', self.comboBox_training_criterion.currentText())

    def set_comboBox_training_optimizer(self):
        self.comboBox_training_optimizer.addItems(self.gui_values.model['training_optimizer_list'])

        index = self.comboBox_training_optimizer.findText(self.app_model.model['training_optimizer'],
                                                          Qt.MatchFixedString)
        if index >= 0:
            self.comboBox_training_optimizer.setCurrentIndex(index)

        self.comboBox_training_optimizer.activated.connect(self.comboBox_training_optimizer_activated)

    def comboBox_training_optimizer_activated(self):
        self.app_model.set_to_model('training_optimizer', self.comboBox_training_optimizer.currentText())

    def set_comboBox_training_scheduler(self):
        self.comboBox_training_scheduler.addItems(self.gui_values.model['training_scheduler_list'])

        index = self.comboBox_training_scheduler.findText(self.app_model.model['training_scheduler'],
                                                          Qt.MatchFixedString)
        if index >= 0:
            self.comboBox_training_scheduler.setCurrentIndex(index)

        self.comboBox_training_scheduler.activated.connect(self.comboBox_training_scheduler_activated)

    def comboBox_training_scheduler_activated(self):
        self.app_model.set_to_model('training_scheduler', self.comboBox_training_scheduler.currentText())

    ###
    # doublespinbox
    ###
    def set_doubleSpinBox_training_dropout(self):
        self.doubleSpinBox_training_dropout.setValue(self.app_model.model['training_dropout'])
        self.doubleSpinBox_training_dropout.valueChanged.connect(self.doubleSpinBox_training_dropout_changed)

    def doubleSpinBox_training_dropout_changed(self, value):
        self.app_model.set_to_model('training_dropout', value)

    def set_doubleSpinBox_training_learning_rate(self):
        self.doubleSpinBox_training_learning_rate.setValue(self.app_model.model['training_learning_rate'])
        self.doubleSpinBox_training_learning_rate.valueChanged.connect(
            self.doubleSpinBox_training_learning_rate_value_changed)

    def doubleSpinBox_training_learning_rate_value_changed(self, value):
        self.app_model.set_to_model('training_learning_rate', value)

    def set_doubleSpinBox_training_lr_gamma(self):
        self.doubleSpinBox_training_lr_gamma.setValue(self.app_model.model['training_lr_gamma'])
        self.doubleSpinBox_training_lr_gamma.valueChanged.connect(self.doubleSpinBox_training_lr_gamma_value_changed)

    def doubleSpinBox_training_lr_gamma_value_changed(self, value):
        self.app_model.set_to_model('training_lr_gamma', value)

    def set_doubleSpinBox_training_lr_step_size(self):
        self.doubleSpinBox_training_lr_step_size.setValue(self.app_model.model['training_lr_step_size'])
        self.doubleSpinBox_training_lr_step_size.valueChanged.connect(
            self.doubleSpinBox_training_lr_step_size_value_changed)

    def doubleSpinBox_training_lr_step_size_value_changed(self, value):
        self.app_model.set_to_model('training_lr_step_size', value)

    def set_doubleSpinBox_training_momentum(self):
        self.doubleSpinBox_training_momentum.setValue(self.app_model.model['training_momentum'])
        self.doubleSpinBox_training_momentum.valueChanged.connect(self.doubleSpinBox_training_momentum_value_changed)

    def doubleSpinBox_training_momentum_value_changed(self, value):
        self.app_model.set_to_model('training_momentum', value)

    ###
    # lineedit
    ###
    def set_lineedit_dataset_directory_with_classes(self):
        self.lineedit_dataset_directory_with_classes.setText(self.app_model.model['dataset_directory_with_classes'])

    def set_lineedit_dataset_train_test_valid_directory(self):
        self.lineedit_dataset_train_test_valid_directory.setText(
            AppModel.get_instance().model['dataset_train_test_valid_directory'])

    def set_lineedit_testing_saved_models_directory(self):
        path = AppModel.get_instance().model['testing_saved_models_directory']
        self.lineedit_testing_saved_models_directory.setText(path)
        models_in_dir = Utils.list_files_with_extensions(path, ['.pth'])
        self.listWidget_testing_saved_models.addItems(models_in_dir)

    def set_lineedit_testing_dataset_test_directory(self):
        path = AppModel.get_instance().model['testing_dataset_test_directory']
        self.lineedit_testing_dataset_test_directory.setText(path)

    def set_lineedit_training_model_output_directory(self):
        self.lineedit_training_model_output_directory.setText(
            AppModel.get_instance().model['training_model_output_directory'])

    def set_lineedit_training_evaluation_directory(self):
        self.lineedit_training_evaluation_directory.setText(
            AppModel.get_instance().model['training_evaluation_directory'])

    ###
    # listwidget
    ###
    def set_listWidget_training_cnn_models_to_train(self):
        self.listWidget_training_cnn_models_to_train.addItems(self.gui_values.model['train_cnn_list'])
        self.listWidget_training_cnn_models_to_train.setSelectionMode(QListWidget.ExtendedSelection)
        self.listWidget_training_cnn_models_to_train_preselect_items()
        self.listWidget_training_cnn_models_to_train.itemSelectionChanged.connect(
            self.listWidget_training_cnn_models_to_train_item_selection_changed)

    def listWidget_training_cnn_models_to_train_item_selection_changed(self):
        selected_items = [item.text() for item in self.listWidget_training_cnn_models_to_train.selectedItems()]
        self.app_model.set_to_model('training_cnn_models_to_train', selected_items)

    def listWidget_training_cnn_models_to_train_preselect_items(self):
        training_cnn_models_to_train = self.app_model.model['training_cnn_models_to_train']

        for model_name in training_cnn_models_to_train:
            matching_items = self.listWidget_training_cnn_models_to_train.findItems(model_name, Qt.MatchExactly)
            for item in matching_items:
                item.setSelected(True)

    def set_listWidget_testing_dataset_class_name(self):
        self.listWidget_testing_dataset_class_name.addItems(
            Utils.list_folders(self.app_model.model['testing_dataset_test_directory']))
        self.listWidget_testing_dataset_class_name.setSelectionMode(QListWidget.ExtendedSelection)
        self.listWidget_testing_dataset_class_name.itemSelectionChanged.connect(
            self.listWidget_testing_dataset_class_name_selection_changed)

    def listWidget_testing_dataset_class_name_selection_changed(self):
        selected_items = [item.text() for item in self.listWidget_testing_dataset_class_name.selectedItems()]
        self.app_model.set_to_model('testing_dataset_class_name_selected', selected_items)

    def set_listWidget_testing_saved_models(self):
        self.listWidget_testing_saved_models.setSelectionMode(QListWidget.ExtendedSelection)
        self.listWidget_testing_saved_models.itemSelectionChanged.connect(
            self.listWidget_testing_saved_models_selection_changed)

    def listWidget_testing_saved_models_selection_changed(self):
        selected_items = [item.text() for item in self.listWidget_testing_saved_models.selectedItems()]
        self.app_model.set_to_model('testing_saved_models_selected', selected_items)

    ###
    # pushbutton
    ###
    def set_pushButton_dataset_show_analysis(self):
        self.pushButton_dataset_show_analysis.clicked.connect(self.pushButton_dataset_show_analysis_clicked)

    def pushButton_dataset_show_analysis_clicked(self):
        print(DatasetAnalyser.get_dataset_statistic(
            dataset_train_test_valid_directory=self.app_model.model['dataset_train_test_valid_directory'],
            dataset_train_dir_name=self.app_model.model['dataset_train_dir_name'],
            dataset_test_dir_name=self.app_model.model['dataset_test_dir_name'],
            dataset_valid_dir_name=self.app_model.model['dataset_valid_dir_name']
        ))

    def set_pushButton_run_samples_distribution(self):
        self.pushButton_run_samples_distribution.clicked.connect(self.pushButton_run_samples_distribution_clicked)

    def pushButton_run_samples_distribution_clicked(self):
        TrainTestValidDatasetCreator.create_train_test_valid_folders_dataset(
            dataset_directory_with_classes=self.app_model.model.get('dataset_directory_with_classes'),
            dataset_train_test_valid_directory=self.app_model.model.get('dataset_train_test_valid_directory'),
            dataset_is_test_set_enabled=self.app_model.model.get('dataset_is_test_set_enabled'),
            dataset_train_set_percentage=self.app_model.model.get('dataset_train_set_percentage'),
            dataset_test_set_percentage=self.app_model.model.get('dataset_test_set_percentage'),
            dataset_valid_set_percentage=self.app_model.model.get('dataset_valid_set_percentage'),
            dataset_train_dir_name=self.app_model.model.get('dataset_train_dir_name'),
            dataset_test_dir_name=self.app_model.model.get('dataset_test_dir_name'),
            dataset_valid_dir_name=self.app_model.model.get('dataset_valid_dir_name')
        )

    def set_pushButton_training(self):
        self.pushButton_training.clicked.connect(self.pushButton_training_clicked)

    def pushButton_training_clicked(self):
        self.dialog_training = DialogTraining()
        self.dialog_training.show()
        self.dialog_training.train()

    def set_pushButton_data_augmentation_train(self):
        self.pushButton_data_augmentation_train.clicked.connect(self.pushButton_data_augmentation_train_clicked)

    def pushButton_data_augmentation_train_clicked(self):
        dialog = DialogDataAugmentation(self.app_model.model['dataset_data_augmentation_train'],
                                        self.set_data_augmentation_model_train)
        dialog.exec_()

    def set_pushButton_data_augmentation_test(self):
        self.pushButton_data_augmentation_test.clicked.connect(self.pushButton_data_augmentation_test_clicked)

    def pushButton_data_augmentation_test_clicked(self):
        dialog = DialogDataAugmentation(self.app_model.model['dataset_data_augmentation_test'],
                                        self.set_data_augmentation_model_test)
        dialog.exec_()

    def set_pushButton_data_augmentation_valid(self):
        self.pushButton_data_augmentation_valid.clicked.connect(self.pushButton_data_augmentation_valid_clicked)

    def pushButton_data_augmentation_valid_clicked(self):
        dialog = DialogDataAugmentation(self.app_model.model['dataset_data_augmentation_valid'],
                                        self.set_data_augmentation_model_valid)
        dialog.exec_()

    def set_pushButton_testing_run(self):
        self.pushButton_testing_run.clicked.connect(self.pushButton_testing_run_clicked)

    def pushButton_testing_run_clicked(self):
        dataloader_test = DataLoader(
            Dataset.get_dataset(os.path.join(self.app_model.model['dataset_train_test_valid_directory'],
                                             self.app_model.model['dataset_test_dir_name']),
                                self.app_model.model['dataset_data_augmentation_test_enabled'],
                                self.app_model.model['dataset_data_augmentation_test']),
            batch_size=self.app_model.model['training_batch_size'],
            shuffle=True
        )
        # model = PyTorchModels.get_cnn_model('alexnet', 2, False, True)
        # webbrowser.open('C://Users//hutas//OneDrive - Slovenská technická univerzita v Bratislave//Škola')

        PrettyPrintConfusionMatrix.plot(y_true, y_pred)
        ModelEvaluation.plot_roc_curve(model, dataloader_test)
        ModelEvaluation.get_accuracy(dataloader_test, model)
        ModelEvaluation.get_accuracy_classes(dataloader_test, model)
        # ModelEvaluation.visualize_model(model, dataloader_test, 8)

    def set_pushButton_testing_confusion_matrix(self):
        self.pushButton_testing_confusion_matrix.clicked.connect(self.pushButton_testing_confusion_matrix_clicked)

    def pushButton_testing_confusion_matrix_clicked(self):
        dataloader_test = DataLoader(
            Dataset.get_dataset(self.app_model.model['testing_dataset_test_directory'],
                                self.app_model.model['dataset_data_augmentation_test_enabled'],
                                self.app_model.model['dataset_data_augmentation_test']),
            batch_size=self.app_model.model['training_batch_size'],
            shuffle=True
        )

        for model_id in self.app_model.model['testing_saved_models_selected']:
            model = ModelIO.load(os.path.join(self.app_model.model['testing_saved_models_directory'], model_id))
            y_true, y_pred, probabilities = ModelEvaluation.get_predictions(model, dataloader_test)
            PrettyPrintConfusionMatrix.plot(y_true, y_pred, classes=dataloader_test.dataset.classes)

            ModelEvaluation.get_accuracy(dataloader_test, model)
            ModelEvaluation.get_accuracy_classes(dataloader_test, model)

    def set_pushButton_testing_confusion_matrix_class(self):
        self.pushButton_testing_confusion_matrix.clicked.connect(self.pushButton_testing_confusion_matrix_clicked)

    def pushButton_testing_confusion_matrix_clicked_class(self):
        print('pushButton_testing_confusion_matrix_clicked_class')

    def set_pushButton_testing_roc_curve(self):
        self.pushButton_testing_roc_curve.clicked.connect(self.pushButton_testing_roc_curve_clicked)

    def pushButton_testing_roc_curve_clicked(self):
        dataloader_test = DataLoader(
            Dataset.get_dataset(os.path.join(self.app_model.model['dataset_train_test_valid_directory'],
                                             self.app_model.model['dataset_test_dir_name']),
                                self.app_model.model['dataset_data_augmentation_test_enabled'],
                                self.app_model.model['dataset_data_augmentation_test']),
            batch_size=self.app_model.model['training_batch_size'],
            shuffle=True
        )

        for model_id in self.app_model.model['testing_saved_models_selected']:

            model = ModelIO.load(os.path.join(self.app_model.model['testing_saved_models_directory'], model_id))
            ModelEvaluation.plot_roc_curve(model, dataloader_test)

            print('pushButton_testing_roc_curve_clicked')

    def set_pushButton_testing_top_k_accuracy(self):
        self.pushButton_testing_top_k_accuracy.clicked.connect(self.pushButton_testing_top_k_accuracy_clicked)

    def pushButton_testing_top_k_accuracy_clicked(self):
        print('pushButton_testing_top_k_accuracy_clicked')

    # ###
    # spinbox
    ###
    def set_spinbox_dataset_train_set_percentage(self):
        self.spinbox_dataset_train_set_percentage.setValue(self.app_model.model['dataset_train_set_percentage'])
        self.spinbox_dataset_train_set_percentage.valueChanged.connect(
            self.spinbox_dataset_train_set_percentage_value_changed)

    def spinbox_dataset_train_set_percentage_value_changed(self, value):
        self.app_model.set_to_model('dataset_train_set_percentage', value)

    def set_spinbox_dataset_test_set_percentage(self):
        self.spinbox_dataset_test_set_percentage.setValue(self.app_model.model['dataset_test_set_percentage'])
        self.spinbox_dataset_test_set_percentage.valueChanged.connect(
            self.spinbox_dataset_test_set_percentage_value_changed)

    def spinbox_dataset_test_set_percentage_value_changed(self, value):
        self.app_model.set_to_model('dataset_test_set_percentage', value)

    def set_spinbox_dataset_valid_set_percentage(self):
        self.spinbox_dataset_valid_set_percentage.setValue(self.app_model.model['dataset_valid_set_percentage'])
        self.spinbox_dataset_valid_set_percentage.valueChanged.connect(
            self.spinbox_dataset_valid_set_percentage_value_changed)

    def spinbox_dataset_valid_set_percentage_value_changed(self, value):
        self.app_model.set_to_model('dataset_valid_set_percentage', value)

    def set_spinBox_training_batch_size(self):
        self.spinBox_training_batch_size.setValue(self.app_model.model['training_batch_size'])
        self.spinBox_training_batch_size.valueChanged.connect(self.spinBox_training_batch_size_value_changed)

    def spinBox_training_batch_size_value_changed(self, value):
        self.app_model.set_to_model('training_batch_size', value)

    def set_spinBox_training_epochs_early_stopping(self):
        self.spinBox_training_epochs_early_stopping.setValue(self.app_model.model['training_epochs_early_stopping'])
        self.spinBox_training_epochs_early_stopping.valueChanged.connect(
            self.spinBox_training_epochs_early_stopping_value_changed)

    def spinBox_training_epochs_early_stopping_value_changed(self, value):
        self.app_model.set_to_model('training_epochs_early_stopping', value)

    def set_spinBox_training_epochs_count(self):
        self.spinBox_training_epochs_count.setValue(self.app_model.model['training_epochs_count'])
        self.spinBox_training_epochs_count.valueChanged.connect(self.spinBox_training_epochs_count_value_changed)

    def spinBox_training_epochs_count_value_changed(self, value):
        self.app_model.set_to_model('training_epochs_count', value)

    ###
    # toolbutton
    ###
    def set_toolbutton_dataset_directory_with_classes(self):
        self.toolbutton_dataset_directory_with_classes.clicked.connect(
            self.toolbutton_dataset_directory_with_classes_clicked)

    def toolbutton_dataset_directory_with_classes_clicked(self):
        path = str(QFileDialog.getExistingDirectory(self, "Select Directory",
                                                    self.app_model.model['dataset_directory_with_classes']))
        self.app_model.set_to_model('dataset_directory_with_classes', path)
        self.lineedit_dataset_directory_with_classes.setText(path)

    def set_toolbutton_dataset_train_test_valid_directory(self):
        self.toolbutton_dataset_train_test_valid_directory.clicked.connect(
            self.toolbutton_dataset_train_test_valid_directory_clicked)

    def toolbutton_dataset_train_test_valid_directory_clicked(self):
        path = str(QFileDialog.getExistingDirectory(self, "Select Directory",
                                                    self.app_model.model['dataset_train_test_valid_directory']))
        self.app_model.set_to_model('dataset_train_test_valid_directory', path)
        self.lineedit_dataset_train_test_valid_directory.setText(path)

    def set_toolbutton_training_evaluation_directory(self):
        self.toolbutton_training_evaluation_directory.clicked.connect(
            self.toolbutton_training_evaluation_directory_clicked)

    def toolbutton_training_evaluation_directory_clicked(self):
        path = str(QFileDialog.getExistingDirectory(self, "Select Directory",
                                                    self.app_model.model['training_evaluation_directory']))
        self.app_model.set_to_model('training_evaluation_directory', path)
        self.lineedit_training_evaluation_directory.setText(path)

    def set_toolbutton_testing_dataset_test_directory(self):
        self.toolbutton_testing_dataset_test_directory.clicked.connect(
            self.toolbutton_testing_dataset_test_directory_clicked)

    def toolbutton_testing_dataset_test_directory_clicked(self):
        path = str(QFileDialog.getExistingDirectory(self, "Select Directory",
                                                    self.app_model.model['testing_dataset_test_directory']))
        self.app_model.set_to_model('testing_dataset_test_directory', path)
        self.lineedit_testing_dataset_test_directory.setText(path)
        self.listWidget_testing_dataset_class_name.clear()
        self.listWidget_testing_dataset_class_name.addItems(Utils.list_folders(path))

    def set_toolbutton_testing_saved_models_directory(self):
        self.toolbutton_testing_saved_models_directory.clicked.connect(
            self.toolbutton_testing_saved_models_directory_clicked)

    def toolbutton_testing_saved_models_directory_clicked(self):
        path = str(QFileDialog.getExistingDirectory(self, "Select Directory",
                                                    self.app_model.model['testing_saved_models_directory']))
        self.app_model.set_to_model('testing_saved_models_directory', path)
        self.lineedit_testing_saved_models_directory.setText(path)
        models_in_dir = Utils.list_files_with_extensions(path, ['.pth'])
        self.listWidget_testing_saved_models.clear()
        self.listWidget_testing_saved_models.addItems(models_in_dir)

    def set_toolbutton_training_model_output_directory(self):
        self.toolbutton_training_model_output_directory.clicked.connect(
            self.toolbutton_training_model_output_directory_clicked)

    def toolbutton_training_model_output_directory_clicked(self):
        path = str(QFileDialog.getExistingDirectory(self, "Select Directory",
                                                    self.app_model.model['training_model_output_directory']))
        self.app_model.set_to_model('training_model_output_directory', path)
        self.lineedit_training_model_output_directory.setText(path)

    ###
    # helpers
    ###
    def set_data_augmentation_model_train(self, data_augmentation_model):
        self.app_model.set_to_model('dataset_data_augmentation_train', data_augmentation_model)

    def set_data_augmentation_model_test(self, data_augmentation_model):
        self.app_model.set_to_model('dataset_data_augmentation_test', data_augmentation_model)

    def set_data_augmentation_model_valid(self, data_augmentation_model):
        self.app_model.set_to_model('dataset_data_augmentation_valid', data_augmentation_model)
