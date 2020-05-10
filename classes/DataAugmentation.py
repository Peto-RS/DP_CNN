from torchvision import transforms


class DataAugmentation:
    @staticmethod
    def get_transform_object_from_data_augmentation_model(data_augmentation_model):
        data_augmentation_operations = []

        if data_augmentation_model.resize_enabled:
            data_augmentation_operations.append(DataAugmentation.get_transform_resize(data_augmentation_model))

        if data_augmentation_model.random_resized_crop_enabled:
            data_augmentation_operations.append(
                DataAugmentation.get_transform_random_resized_crop(data_augmentation_model))

        if data_augmentation_model.random_rotation_enabled:
            data_augmentation_operations.append(DataAugmentation.get_transform_random_rotation(data_augmentation_model))

        if data_augmentation_model.color_jitter_enabled:
            data_augmentation_operations.append(DataAugmentation.get_transform_color_jitter(data_augmentation_model))

        if data_augmentation_model.random_horizontal_flip_enabled:
            data_augmentation_operations.append(
                DataAugmentation.get_transform_random_horizontal_flip(data_augmentation_model))

        if data_augmentation_model.random_vertical_flip_enabled:
            data_augmentation_operations.append(
                DataAugmentation.get_transform_random_vertical_flip(data_augmentation_model))

        if data_augmentation_model.center_crop_enabled:
            data_augmentation_operations.append(
                DataAugmentation.get_transform_center_crop(data_augmentation_model),
            )

        if data_augmentation_model.grayscale_enabled:
            data_augmentation_operations.append(
                DataAugmentation.get_transform_grayscale(data_augmentation_model)
            )

        data_augmentation_operations.append(transforms.ToTensor())

        if data_augmentation_model.normalize_enabled:
            data_augmentation_operations.append(
                DataAugmentation.get_transform_normalize(data_augmentation_model)
            )

        return transforms.Compose(data_augmentation_operations)

    @staticmethod
    def get_transform_resize(data_augmentation_model):
        if data_augmentation_model.resize_image_size_single_enabled:
            return transforms.Resize(size=data_augmentation_model.resize_image_size_single)
        elif data_augmentation_model.resize_image_size_wh_enabled:
            return transforms.Resize(
                size=(data_augmentation_model.resize_image_height, data_augmentation_model.resize_image_width))

    @staticmethod
    def get_transform_random_rotation(data_augmentation_model):
        return transforms.RandomRotation(degrees=data_augmentation_model.random_rotation_angle)

    @staticmethod
    def get_transform_normalize(data_augmentation_model):
        return transforms.Normalize([
            data_augmentation_model.normalize_mean1,
            data_augmentation_model.normalize_mean2,
            data_augmentation_model.normalize_mean3
        ], [
            data_augmentation_model.normalize_std1,
            data_augmentation_model.normalize_std2,
            data_augmentation_model.normalize_std3
        ])

    @staticmethod
    def get_transform_color_jitter(data_augmentation_model):
        return transforms.ColorJitter()

    @staticmethod
    def get_transform_random_horizontal_flip(data_augmentation_model):
        return transforms.RandomHorizontalFlip(p=data_augmentation_model.random_horizontal_flip_probability)

    @staticmethod
    def get_transform_random_vertical_flip(data_augmentation_model):
        return transforms.RandomVerticalFlip(p=data_augmentation_model.random_vertical_flip_probability)

    @staticmethod
    def get_transform_center_crop(data_augmentation_model):
        if data_augmentation_model.center_crop_image_size_single_enabled:
            return transforms.CenterCrop(size=data_augmentation_model.center_crop_image_size_single)
        elif data_augmentation_model.center_crop_image_size_wh_enabled:
            return transforms.CenterCrop(size=(
                data_augmentation_model.center_crop_image_height, data_augmentation_model.center_crop_image_width))

    @staticmethod
    def get_transform_random_resized_crop(data_augmentation_model):
        scale = (data_augmentation_model.random_resized_crop_scale1, data_augmentation_model.random_resized_crop_scale2)
        ratio = (data_augmentation_model.random_resized_crop_ratio1, data_augmentation_model.random_resized_crop_ratio2)

        if data_augmentation_model.random_resized_crop_image_size_single_enabled:
            return transforms.RandomResizedCrop(
                size=data_augmentation_model.random_resized_crop_image_size_single,
                scale=scale,
                ratio=ratio
            )
        elif data_augmentation_model.random_resized_crop_image_size_wh_enabled:
            return transforms.RandomResizedCrop(size=(data_augmentation_model.random_resized_crop_image_height,
                                                      data_augmentation_model.random_resized_crop_image_width),
                                                scale=scale,
                                                ratio=ratio)

    @staticmethod
    def get_transform_grayscale(data_augmentation_model):
        return transforms.Grayscale()
