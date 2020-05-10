import copy
from datetime import datetime

import numpy as np
import os
import time
import torch
import torch.nn as nn

from classes.ModelEvaluation import ModelEvaluation
from classes.ModelIO import ModelIO
from classes.Utils import Utils
from enums.PyTorchModelsEnum import PyTorchModelsEnum
from old.GlobalSettings import GlobalSettings
from classes.PyTorchModels import PyTorchModels
from classes.nn.Criterion import Criterion
from classes.nn.Optimizer import Optimizer
from classes.nn.Scheduler import Scheduler


class ModelTraining:
    @staticmethod
    def train_models(dataset_dataloader, dataset_test_dir_name, dataset_train_dir_name, dataset_valid_dir_name,
                     training_cnn_models_to_train, training_criterion, training_dropout, training_epochs_early_stopping,
                     training_epochs_count,
                     training_feature_extract, training_learning_rate, training_lr_gamma, training_lr_step_size,
                     training_model_output_directory, training_momentum, training_optimizer,
                     training_save_best_model_enabled, training_scheduler, training_use_gpu,
                     training_use_early_stopping,
                     training_use_pretrained_models, training_use_softmax, signals):
        for i, model_id in enumerate(training_cnn_models_to_train):
            signals['label_training_model_name_text_changed'].emit(model_id)
            signals['console_append'].emit('----------\n' + model_id + '\n----------')

            model, accuracy_loss_history = ModelTraining.train_model(model_id=model_id,
                                                                     dataset_dataloader=dataset_dataloader,
                                                                     dataset_test_dir_name=dataset_test_dir_name,
                                                                     dataset_train_dir_name=dataset_train_dir_name,
                                                                     dataset_valid_dir_name=dataset_valid_dir_name,
                                                                     training_criterion=training_criterion,
                                                                     training_dropout=training_dropout,
                                                                     training_epochs_count=training_epochs_count,
                                                                     training_epochs_early_stopping=training_epochs_early_stopping,
                                                                     training_use_gpu=training_use_gpu,
                                                                     training_use_early_stopping=training_use_early_stopping,
                                                                     training_feature_extract=training_feature_extract,
                                                                     training_learning_rate=training_learning_rate,
                                                                     training_lr_gamma=training_lr_gamma,
                                                                     training_lr_step_size=training_lr_step_size,
                                                                     training_momentum=training_momentum,
                                                                     training_optimizer=training_optimizer,
                                                                     training_scheduler=training_scheduler,
                                                                     training_use_pretrained_models=training_use_pretrained_models,
                                                                     training_use_softmax=training_use_softmax,
                                                                     signals=signals
                                                                     )

            if training_save_best_model_enabled:
                path = Utils.create_folder_if_not_exists(training_model_output_directory)
                ModelIO.save(model=model, model_id=model_id, output_dir=path,
                             classes=dataset_dataloader[dataset_train_dir_name].dataset.classes,
                             feature_extract=training_feature_extract,
                             use_pretrained=training_use_pretrained_models, training_dropout=training_dropout,
                             training_use_softmax=training_use_softmax)

            signals['plot_train_valid_acc_loss_graph'].emit(accuracy_loss_history, model_id)
            signals['progressBar_training_set_value_changed'].emit((i + 1) / len(training_cnn_models_to_train) * 100)

    @staticmethod
    def train_model(model_id, dataset_dataloader, dataset_train_dir_name, dataset_test_dir_name,
                    dataset_valid_dir_name, training_criterion, training_dropout, training_learning_rate,
                    training_lr_gamma,
                    training_lr_step_size, training_momentum, training_optimizer, training_scheduler, training_use_gpu,
                    training_feature_extract, training_epochs_count, training_epochs_early_stopping,
                    training_use_early_stopping, training_use_pretrained_models, training_use_softmax, signals):
        num_classes = len(dataset_dataloader[dataset_train_dir_name].dataset.classes)
        device = ModelTraining.get_device(training_use_gpu=training_use_gpu)
        model = PyTorchModels.get_cnn_model(model_id=model_id, num_classes=num_classes,
                                            feature_extract=training_feature_extract,
                                            use_pretrained=training_use_pretrained_models,
                                            training_dropout=training_dropout,
                                            training_use_softmax=training_use_softmax)
        model = model.to(device)
        criterion = Criterion.get_criterion(training_criterion_name=training_criterion)
        optimizer = Optimizer.get_optimizer(model, training_feature_extract=training_feature_extract,
                                            training_optimizer_name=training_optimizer,
                                            training_learning_rate=training_learning_rate,
                                            training_momentum=training_momentum)
        scheduler = Scheduler.get_scheduler(optimizer=optimizer, training_scheduler_name=training_scheduler,
                                            training_lr_gamma=training_lr_gamma,
                                            training_lr_step_size=training_lr_step_size)
        model, accuracy_loss_history = ModelTraining.train_fnc(criterion=criterion,
                                                               dataset_dataloader=dataset_dataloader,
                                                               dataset_train_dir_name=dataset_train_dir_name,
                                                               dataset_valid_dir_name=dataset_valid_dir_name,
                                                               device=device,
                                                               model_id=model_id,
                                                               model=model, optimizer=optimizer, scheduler=scheduler,
                                                               training_epochs_count=training_epochs_count,
                                                               training_epochs_early_stopping=training_epochs_early_stopping,
                                                               training_use_early_stopping=training_use_early_stopping,
                                                               signals=signals)

        return model, accuracy_loss_history

    @staticmethod
    def train_fnc(criterion, dataset_dataloader, dataset_train_dir_name, dataset_valid_dir_name,
                  device, model_id, model, optimizer, scheduler,
                  training_epochs_count, training_epochs_early_stopping, training_use_early_stopping, signals):
        accuracy_loss_history = []
        train_loss, valid_loss, train_acc, valid_acc = 0, 0, 0, 0
        best_acc = 0
        best_model_wts = copy.deepcopy(model.state_dict())
        dataset_sizes = {x: len(dataset_dataloader[x].dataset) for x in
                         [dataset_train_dir_name, dataset_valid_dir_name]}
        epochs_no_improve = 0
        epoch_since = None
        since = time.time()
        train_phase = dataset_train_dir_name
        valid_phase = dataset_valid_dir_name

        for epoch in range(training_epochs_count):
            epoch_since = time.time()
            signals['console_append'].emit('Epoch: {}/{}'.format(epoch + 1, training_epochs_count))
            signals['label_epoch_current_total_text_changed'].emit('{}/{}'.format(epoch + 1, training_epochs_count))

            for phase in [train_phase, valid_phase]:
                if phase == train_phase:
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_corrects = 0

                for inputs, labels in dataset_dataloader[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == train_phase):
                        if PyTorchModelsEnum.INCEPTION_V3 == model_id and phase == 'train':
                            outputs, aux_outputs = model(inputs)
                            loss1 = criterion(outputs, labels)
                            loss2 = criterion(aux_outputs, labels)
                            loss = loss1 + 0.4 * loss2
                        else:
                            outputs = model(inputs)
                            loss = criterion(outputs, labels)

                        _, preds = torch.max(outputs, 1)

                        # backward + optimize only if in training phase
                        if phase == train_phase:
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                if phase == train_phase:
                    scheduler.step()
                    train_acc = epoch_acc
                    train_loss = epoch_loss

                signals['console_append'].emit('{} Loss: {:.4f} Accuracy: {:.4f}'.format(phase, epoch_loss, epoch_acc))

                if phase == valid_phase:
                    valid_acc = epoch_acc
                    valid_loss = epoch_loss

                    if epoch_acc > best_acc:
                        best_acc = epoch_acc
                        best_model_wts = copy.deepcopy(model.state_dict())
                        epochs_no_improve = 0
                    else:
                        epochs_no_improve += 1

                        if training_use_early_stopping and epochs_no_improve >= training_epochs_early_stopping:
                            time_elapsed = time.time() - since

                            signals['console_append'].emit('\nxxxxx\nEarly stopping triggered!\nxxxxx\n')
                            signals['console_append'].emit(
                                'Training duration: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
                            signals['console_append'].emit('Best validation accuracy: {:4f}'.format(best_acc))
                            accuracy_loss_history.append([train_acc, train_loss, valid_acc, valid_loss])
                            model.load_state_dict(best_model_wts)
                            return model, accuracy_loss_history

            accuracy_loss_history.append([train_acc, train_loss, valid_acc, valid_loss])

            time_elapsed_epoch = time.time() - epoch_since
            signals['console_append'].emit(
                'Duration: {:.0f}m {:.0f}s'.format(time_elapsed_epoch // 60, time_elapsed_epoch % 60))
            signals['console_append'].emit('\n')

        time_elapsed = time.time() - since
        signals['console_append'].emit(
            'Training duration: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        signals['console_append'].emit('Best validation accuracy: {:4f}'.format(best_acc))
        signals['console_append'].emit('\n')

        model.load_state_dict(best_model_wts)
        return model, accuracy_loss_history

    @staticmethod
    def get_device(training_use_gpu):
        if torch.cuda.is_available() and training_use_gpu:
            return torch.device("cuda:0")

        return torch.device("cpu")
