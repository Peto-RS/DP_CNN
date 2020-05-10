import imagehash
import os
import random
from shutil import copyfile
import torch

from classes.Utils import Utils


class TrainTestValidDatasetCreator:
    @staticmethod
    def create_train_test_valid_folders_dataset(dataset_directory_with_classes, dataset_train_test_valid_directory,
                                                dataset_is_test_set_enabled, dataset_train_set_percentage,
                                                dataset_test_set_percentage, dataset_valid_set_percentage,
                                                dataset_train_dir_name, dataset_test_dir_name, dataset_valid_dir_name):
        dataset_folders_to_create = [dataset_train_dir_name, dataset_test_dir_name]
        if dataset_is_test_set_enabled:
            dataset_folders_to_create.append(dataset_valid_dir_name)

        dataset_classes = os.listdir(dataset_directory_with_classes)
        TrainTestValidDatasetCreator.create_train_test_valid_folders(dataset_train_test_valid_directory=dataset_train_test_valid_directory,
                                                                     dataset_folders_to_create=dataset_folders_to_create,
                                                                     dataset_classes=dataset_classes)

        files = None
        train_set_count = None
        test_set_count = None
        valid_set_count = None
        train_set_filenames = None
        test_set_filenames = None
        valid_set_filenames = None
        for dataset_classname in dataset_classes:
            class_folder_path = os.path.join(dataset_directory_with_classes, dataset_classname)
            files = os.listdir(class_folder_path)
            files_count = len(files)
            train_set_count = int((dataset_train_set_percentage / 100) * files_count)
            test_set_count = int((dataset_test_set_percentage / 100) * files_count)
            if dataset_is_test_set_enabled:
                valid_set_count = int((dataset_valid_set_percentage / 100) * files_count)

            # TODO
            print("CLASS NAME: {} FILES COUNT: {}".format(dataset_classname, files_count))
            print("TRAIN: {} TEST: {} VAL: {}".format(train_set_count, test_set_count, valid_set_count))

            random.shuffle(files)
            train_set_filenames = files[:train_set_count]
            test_set_filenames = files[train_set_count: train_set_count + test_set_count]

            if dataset_is_test_set_enabled:
                valid_set_filenames = files[train_set_count + test_set_count: files_count - 1]

            TrainTestValidDatasetCreator.copy_files_between_folders(
                list_of_names=train_set_filenames,
                src_folder=class_folder_path,
                dst_folder=os.path.join(dataset_train_test_valid_directory, dataset_train_dir_name, dataset_classname)
            )

            TrainTestValidDatasetCreator.copy_files_between_folders(
                list_of_names=test_set_filenames,
                src_folder=class_folder_path,
                dst_folder=os.path.join(dataset_train_test_valid_directory, dataset_test_dir_name, dataset_classname)
            )

            if dataset_is_test_set_enabled:
                TrainTestValidDatasetCreator.copy_files_between_folders(
                    list_of_names=valid_set_filenames,
                    src_folder=class_folder_path,
                    dst_folder=os.path.join(dataset_train_test_valid_directory, dataset_valid_dir_name,
                                            dataset_classname)
                )

    @staticmethod
    def create_train_test_valid_folders(dataset_train_test_valid_directory, dataset_folders_to_create, dataset_classes):
        for dirname in dataset_folders_to_create:
            Utils.create_folder_if_not_exists(os.path.join(dataset_train_test_valid_directory, dirname))

            for class_name in dataset_classes:
                Utils.create_folder_if_not_exists(os.path.join(dataset_train_test_valid_directory, dirname, class_name))

    @staticmethod
    def copy_files_between_folders(list_of_names, src_folder, dst_folder):
        for filename in list_of_names:
            copyfile(os.path.join(src_folder, filename), os.path.join(dst_folder, filename))

    @staticmethod
    def move_files_between_folders(list_of_names, src_folder, dst_folder):
        for filename in list_of_names:
            os.rename(os.path.join(src_folder, filename), os.path.join(dst_folder, filename))
