import os
from torchvision import datasets

from classes.DataAugmentation import DataAugmentation
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class Dataset:
    @staticmethod
    def create_dataloader_for_neural_network(dataset_train_test_valid_directory, dataset_train_dir_name,
                                             dataset_test_dir_name,
                                             dataset_is_test_set_enabled, dataset_valid_dir_name,
                                             dataset_data_augmentation_train_enabled, dataset_data_augmentation_train,
                                             dataset_data_augmentation_test_enabled, dataset_data_augmentation_test,
                                             dataset_data_augmentation_valid_enabled, dataset_data_augmentation_valid,
                                             training_batch_size):
        dataset = {
            dataset_train_dir_name:
                Dataset.get_dataset(os.path.join(dataset_train_test_valid_directory, dataset_train_dir_name),
                                    dataset_data_augmentation_train_enabled, dataset_data_augmentation_train),
            dataset_valid_dir_name:
                Dataset.get_dataset(os.path.join(dataset_train_test_valid_directory, dataset_valid_dir_name),
                                    dataset_data_augmentation_valid_enabled, dataset_data_augmentation_valid),
        }

        if dataset_is_test_set_enabled:
            dataset[dataset_test_dir_name] = Dataset.get_dataset(
                os.path.join(dataset_train_test_valid_directory, dataset_test_dir_name),
                dataset_data_augmentation_test_enabled, dataset_data_augmentation_test)

        data_loaders = {
            dataset_train_dir_name: DataLoader(
                dataset[dataset_train_dir_name],
                batch_size=training_batch_size,
                shuffle=True
            ),
            dataset_valid_dir_name: DataLoader(
                dataset[dataset_valid_dir_name],
                batch_size=training_batch_size,
                shuffle=True
            )
        }

        if dataset_is_test_set_enabled:
            data_loaders[dataset_test_dir_name] = DataLoader(
                dataset[dataset_test_dir_name],
                batch_size=training_batch_size,
                shuffle=True
            )

        return dataset, data_loaders

    @staticmethod
    def get_dataset(dataset_path, data_augmentation_enabled, data_augmentation_model):
        transforms_obj = DataAugmentation.get_transform_object_from_data_augmentation_model(
            data_augmentation_model) if data_augmentation_enabled else transforms.Compose([])

        return datasets.ImageFolder(
            root=dataset_path,
            transform=transforms_obj
        )
