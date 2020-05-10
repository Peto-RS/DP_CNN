from PyQt5.QtCore import QThread

from app_model.AppModel import AppModel

from PyQt5 import QtWidgets, uic

from classes.ModelEvaluation import ModelEvaluation
from gui.threads.TrainingThread import TrainingThread


class DialogTraining(QtWidgets.QDialog):
    def __init__(self):
        super(DialogTraining, self).__init__()
        uic.loadUi('gui/DialogTraining.ui', self)

        self.objThread = None
        self.obj = None

        self.init_gui()

        self.label_epoch_current_total = self.findChild(QtWidgets.QLabel, 'label_epoch_current_total')
        self.label_training_model_name = self.findChild(QtWidgets.QLabel, 'label_training_model_name')
        self.progressBar_training = self.findChild(QtWidgets.QProgressBar, 'progressBar_training')
        self.textBrowser_training = self.findChild(QtWidgets.QTextBrowser, 'textBrowser_training')

    def init_gui(self):
        self.set_textBrowser_training()

    def train(self):
        self.objThread = QThread()
        self.obj = TrainingThread(AppModel.get_instance())

        self.obj.console_append.connect(self.textBrowser_training_value_append)
        self.obj.console_clear.connect(self.textBrowser_training_value_clear)
        self.obj.console_replace.connect(self.textBrowser_training_value_replace)
        self.obj.label_epoch_current_total_text_changed.connect(self.label_epoch_current_total_text_changed)
        self.obj.label_training_model_name_text_changed.connect(self.label_training_model_name_text_changed)
        self.obj.plot_train_valid_acc_loss_graph.connect(self.plot_train_valid_acc_loss_graph)
        self.obj.progressBar_training_set_value_changed.connect(self.progressBar_training_set_value)

        self.obj.moveToThread(self.objThread)
        self.obj.finished.connect(self.objThread.quit)
        self.objThread.started.connect(self.obj.run)
        self.objThread.start()

    ###
    # label
    ###
    def label_epoch_current_total_text_changed(self, text):
        self.label_epoch_current_total.setText(text)

    def label_training_model_name_text_changed(self, text):
        self.label_training_model_name.setText(text)

    ###
    # progressBar
    ###
    def progressBar_training_set_value(self, value):
        self.progressBar_training.setValue(value)

    ###
    # textBrowser
    ###
    def set_textBrowser_training(self):
        self.textBrowser_training.setStyleSheet("background-color: black; color: white")

    def textBrowser_training_value_append(self, text):
        self.textBrowser_training.append(text)

    def textBrowser_training_value_clear(self):
        self.textBrowser_training.setText('')

    def textBrowser_training_value_replace(self, text):
        self.textBrowser_training.setText(text)

    @staticmethod
    def plot_train_valid_acc_loss_graph(accuracy_loss_history, model_name):
        ModelEvaluation.train_valid_acc_loss_graph(history=accuracy_loss_history, model_name=model_name,
                                                   save=AppModel.get_instance().model[
                                                       'training_save_train_valid_graph'],
                                                   training_evaluation_directory=AppModel.get_instance().model[
                                                       'training_evaluation_directory'])
