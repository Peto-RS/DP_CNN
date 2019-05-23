import os
import random
from shutil import copyfile

TRAIN_DIR_NAME = "train"
TEST_DIR_NAME = "test"
VALID_DIR_NAME = "valid"


class InputDatasetCreator:
    def __init__(self, path_to_classes, train_set_percent, test_set_percent, val_set_percent, copy_files):
        self.path = path_to_classes
        self.train_set_percent = train_set_percent
        self.test_set_percent = test_set_percent
        self.val_set_percent = val_set_percent
        self.copy_files = copy_files

    def create_test_valid_train_folders_dataset(self):
        #####
        # CREATE FOLDERS
        #####
        dataset_folders_classes = os.listdir(self.path)

        def filter_predicate(x):
            is_dir = os.path.isdir(os.path.join(self.path, x))
            filter_set_folders_name = x not in [TRAIN_DIR_NAME, TEST_DIR_NAME, VALID_DIR_NAME]
            return is_dir and filter_set_folders_name

        dataset_folders_classes = list(filter(filter_predicate, dataset_folders_classes))

        for dirname in [TRAIN_DIR_NAME, TEST_DIR_NAME, VALID_DIR_NAME]:
            self.create_folder_if_not_exists(os.path.join(self.path, dirname))

            for class_name in dataset_folders_classes:
                self.create_folder_if_not_exists(os.path.join(self.path, dirname, class_name))

        #####
        # MOVE OR COPY FILES FROM FOLDERS
        #####
        for class_folder_name in dataset_folders_classes:
            files = os.listdir(os.path.join(self.path, class_folder_name))
            files_count = len(files)
            train_set_count = int((self.train_set_percent / 100) * files_count)
            test_set_count = int((self.test_set_percent / 100) * files_count)
            val_set_count = int((self.val_set_percent / 100) * files_count)

            print("CLASS NAME: {} FILES COUNT: {}".format(class_folder_name, files_count))
            print("TRAIN: {} TEST: {} VAL: {}".format(train_set_count, test_set_count, val_set_count))

            random.shuffle(files)
            train_set_filenames = files[:train_set_count]
            test_set_filenames = files[train_set_count:train_set_count + test_set_count]
            valid_set_filenames = files[
                                  train_set_count + test_set_count:train_set_count + test_set_count + val_set_count]

            class_folder_path = os.path.join(self.path, class_folder_name)

            if not self.copy_files:
                self.move_files_between_folders(
                    train_set_filenames,
                    class_folder_path,
                    os.path.join(self.path, TRAIN_DIR_NAME, class_folder_name)
                )

                self.move_files_between_folders(
                    test_set_filenames,
                    class_folder_path,
                    os.path.join(self.path, TEST_DIR_NAME, class_folder_name)
                )

                self.move_files_between_folders(
                    valid_set_filenames,
                    class_folder_path,
                    os.path.join(self.path, VALID_DIR_NAME, class_folder_name)
                )
            else:
                self.copy_files_between_folders(
                    train_set_filenames,
                    class_folder_path,
                    os.path.join(self.path, TRAIN_DIR_NAME, class_folder_name)
                )

                self.copy_files_between_folders(
                    test_set_filenames,
                    class_folder_path,
                    os.path.join(self.path, TEST_DIR_NAME, class_folder_name)
                )

                self.copy_files_between_folders(
                    valid_set_filenames,
                    class_folder_path,
                    os.path.join(self.path, VALID_DIR_NAME, class_folder_name)
                )

    @staticmethod
    def create_folder_if_not_exists(path):
        if not os.path.exists(path):
            os.mkdir(path)

    @staticmethod
    def move_files_between_folders(list_of_files, src_folder, dst_folder):
        for filename in list_of_files:
            os.rename(os.path.join(src_folder, filename), os.path.join(dst_folder, filename))

    @staticmethod
    def copy_files_between_folders(list_of_files, src_folder, dst_folder):
        for filename in list_of_files:
            copyfile(os.path.join(src_folder, filename), os.path.join(dst_folder, filename))
