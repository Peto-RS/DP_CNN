class DataAugmentationModelTest:
    normalize_enabled = True
    normalize_mean1 = 0.485
    normalize_mean2 = 0.456
    normalize_mean3 = 0.406
    normalize_std1 = 0.229
    normalize_std2 = 0.224
    normalize_std3 = 0.225
    center_crop_enabled = False
    center_crop_image_size_single_enabled = False
    center_crop_image_size_single = 224
    center_crop_image_size_wh_enabled = False
    center_crop_image_width = 224
    center_crop_image_height = 224
    resize_enabled = True
    resize_image_size_single_enabled = False
    resize_image_size_single = 224
    resize_image_size_wh_enabled = True
    resize_image_width = 64
    resize_image_height = 64
    random_horizontal_flip_enabled = False
    random_horizontal_flip_probability = 0.5
    random_vertical_flip_enabled = False
    random_vertical_flip_probability = 0.5
    color_jitter_enabled = False
    random_rotation_enabled = False
    random_rotation_angle = 15
    random_resized_crop_enabled = False
    random_resized_crop_image_size_single_enabled = False
    random_resized_crop_image_size_single = 224
    random_resized_crop_image_size_wh_enabled = False
    random_resized_crop_image_height = 224
    random_resized_crop_image_width = 224
    random_resized_crop_scale1 = 0.8
    random_resized_crop_scale2 = 1
    random_resized_crop_ratio1 = 0.75
    random_resized_crop_ratio2 = 1.3333333333333333
    grayscale_enabled = False
