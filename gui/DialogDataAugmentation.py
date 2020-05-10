from app_model.data_augmentation.DataAugmentationModelTrain import DataAugmentationModelTrain

from PyQt5 import QtWidgets, uic

SPINBOX_MAX_VALUE = 999999999


class DialogDataAugmentation(QtWidgets.QDialog):
    def __init__(self, data_augmentation_model, callback_fnc):
        super(DialogDataAugmentation, self).__init__()
        uic.loadUi('gui/DialogDataAugmentation.ui', self)

        self.data_augmentation_model = data_augmentation_model
        self.callback_fnc = callback_fnc

        self.buttonBox = self.findChild(QtWidgets.QDialogButtonBox, 'buttonBox')
        self.checkBox_normalize_enabled = self.findChild(QtWidgets.QCheckBox, 'checkBox_normalize_enabled')
        self.checkBox_center_crop_enabled = self.findChild(QtWidgets.QCheckBox, 'checkBox_center_crop_enabled')
        self.checkBox_center_crop_image_size_single_enabled = self.findChild(QtWidgets.QCheckBox,
                                                                             'checkBox_center_crop_image_size_single_enabled')
        self.checkBox_center_crop_image_size_wh_enabled = self.findChild(QtWidgets.QCheckBox,
                                                                         'checkBox_center_crop_image_size_wh_enabled')
        self.checkBox_resize_enabled = self.findChild(QtWidgets.QCheckBox, 'checkBox_resize_enabled')
        self.checkBox_resize_image_size_single_enabled = self.findChild(QtWidgets.QCheckBox,
                                                                        'checkBox_resize_image_size_single_enabled')
        self.checkBox_resize_image_size_wh_enabled = self.findChild(QtWidgets.QCheckBox,
                                                                    'checkBox_resize_image_size_wh_enabled')
        self.checkBox_random_horizontal_flip_enabled = self.findChild(QtWidgets.QCheckBox,
                                                                      'checkBox_random_horizontal_flip_enabled')
        self.checkBox_random_vertical_flip_enabled = self.findChild(QtWidgets.QCheckBox,
                                                                    'checkBox_random_vertical_flip_enabled')
        self.checkBox_color_jitter_enabled = self.findChild(QtWidgets.QCheckBox, 'checkBox_color_jitter_enabled')
        self.checkBox_random_rotation_enabled = self.findChild(QtWidgets.QCheckBox, 'checkBox_random_rotation_enabled')
        self.checkBox_random_resized_crop_enabled = self.findChild(QtWidgets.QCheckBox,
                                                                   'checkBox_random_resized_crop_enabled')
        self.checkBox_random_resized_crop_image_size_single_enabled = self.findChild(QtWidgets.QCheckBox,
                                                                                     'checkBox_random_resized_crop_image_size_single_enabled')
        self.checkBox_random_resized_crop_image_size_wh_enabled = self.findChild(QtWidgets.QCheckBox,
                                                                                 'checkBox_random_resized_crop_image_size_wh_enabled')
        self.checkBox_grayscale_enabled = self.findChild(QtWidgets.QCheckBox, 'checkBox_grayscale_enabled')
        self.doubleSpinBox_mean1 = self.findChild(QtWidgets.QDoubleSpinBox, 'doubleSpinBox_mean1')
        self.doubleSpinBox_mean2 = self.findChild(QtWidgets.QDoubleSpinBox, 'doubleSpinBox_mean2')
        self.doubleSpinBox_mean3 = self.findChild(QtWidgets.QDoubleSpinBox, 'doubleSpinBox_mean3')
        self.doubleSpinBox_std1 = self.findChild(QtWidgets.QDoubleSpinBox, 'doubleSpinBox_std1')
        self.doubleSpinBox_std2 = self.findChild(QtWidgets.QDoubleSpinBox, 'doubleSpinBox_std2')
        self.doubleSpinBox_std3 = self.findChild(QtWidgets.QDoubleSpinBox, 'doubleSpinBox_std3')
        self.doubleSpinBox_random_resized_crop_scale1 = self.findChild(QtWidgets.QDoubleSpinBox,
                                                                       'doubleSpinBox_random_resized_crop_scale1')
        self.doubleSpinBox_random_resized_crop_scale2 = self.findChild(QtWidgets.QDoubleSpinBox,
                                                                       'doubleSpinBox_random_resized_crop_scale2')
        self.doubleSpinBox_random_resized_crop_ratio1 = self.findChild(QtWidgets.QDoubleSpinBox,
                                                                       'doubleSpinBox_random_resized_crop_ratio1')
        self.doubleSpinBox_random_resized_crop_ratio2 = self.findChild(QtWidgets.QDoubleSpinBox,
                                                                       'doubleSpinBox_random_resized_crop_ratio2')
        self.doubleSpinBox_random_horizontal_flip_probability = self.findChild(QtWidgets.QDoubleSpinBox,
                                                                               'doubleSpinBox_random_horizontal_flip_probability')
        self.doubleSpinBox_random_vertical_flip_probability = self.findChild(QtWidgets.QDoubleSpinBox,
                                                                             'doubleSpinBox_random_vertical_flip_probability')
        self.spinBox_center_crop_image_size_single = self.findChild(QtWidgets.QSpinBox,
                                                                    'spinBox_center_crop_image_size_single')
        self.spinBox_center_crop_image_width = self.findChild(QtWidgets.QSpinBox, 'spinBox_center_crop_image_width')
        self.spinBox_center_crop_image_height = self.findChild(QtWidgets.QSpinBox, 'spinBox_center_crop_image_height')
        self.spinBox_resize_image_size_single = self.findChild(QtWidgets.QSpinBox, 'spinBox_resize_image_size_single')
        self.spinBox_resize_image_width = self.findChild(QtWidgets.QSpinBox, 'spinBox_resize_image_width')
        self.spinBox_resize_image_height = self.findChild(QtWidgets.QSpinBox, 'spinBox_resize_image_height')
        self.spinBox_random_rotation_angle = self.findChild(QtWidgets.QSpinBox, 'spinBox_random_rotation_angle')
        self.spinBox_random_resized_crop_image_size_single = self.findChild(QtWidgets.QSpinBox,
                                                                            'spinBox_random_resized_crop_image_size_single')
        self.spinBox_random_resized_crop_image_width = self.findChild(QtWidgets.QSpinBox,
                                                                      'spinBox_random_resized_crop_image_width')
        self.spinBox_random_resized_crop_image_height = self.findChild(QtWidgets.QSpinBox,
                                                                       'spinBox_random_resized_crop_image_height')

        self.init_gui()

    def init_gui(self):
        self.set_buttonBox()
        self.set_checkBox_normalize_enabled()
        self.set_checkBox_center_crop_enabled()
        self.set_checkBox_center_crop_image_size_single_enabled()
        self.set_checkBox_center_crop_image_size_wh_enabled()
        self.set_checkBox_resize_enabled()
        self.set_checkBox_resize_image_size_single_enabled()
        self.set_checkBox_resize_image_size_wh_enabled()
        self.set_checkBox_random_horizontal_flip_enabled()
        self.set_checkBox_random_vertical_flip_enabled()
        self.set_checkBox_color_jitter_enabled()
        self.set_checkBox_random_rotation_enabled()
        self.set_checkBox_random_resized_crop_enabled()
        self.set_checkBox_random_resized_crop_image_size_single_enabled()
        self.set_checkBox_random_resized_crop_image_size_wh_enabled()
        self.set_checkBox_grayscale_enabled()
        self.set_doubleSpinBox_mean1()
        self.set_doubleSpinBox_mean2()
        self.set_doubleSpinBox_mean3()
        self.set_doubleSpinBox_std1()
        self.set_doubleSpinBox_std2()
        self.set_doubleSpinBox_std3()
        self.set_doubleSpinBox_random_resized_crop_scale1()
        self.set_doubleSpinBox_random_resized_crop_scale2()
        self.set_doubleSpinBox_random_resized_crop_ratio1()
        self.set_doubleSpinBox_random_resized_crop_ratio2()
        self.set_doubleSpinBox_random_horizontal_flip_probability()
        self.set_doubleSpinBox_random_vertical_flip_probability()
        self.set_spinBox_center_crop_image_size_single()
        self.set_spinBox_center_crop_image_width()
        self.set_spinBox_center_crop_image_height()
        self.set_spinBox_resize_image_size_single()
        self.set_spinBox_resize_image_width()
        self.set_spinBox_resize_image_height()
        self.set_spinBox_random_rotation_angle()
        self.set_spinBox_random_resized_crop_image_size_single()
        self.set_spinBox_random_resized_crop_image_width()
        self.set_spinBox_random_resized_crop_image_height()

    def set_buttonBox(self):
        self.buttonBox.accepted.connect(self.set_buttonBox_clicked)

    def set_buttonBox_clicked(self):
        data_augmentation_model = DataAugmentationModelTrain()
        data_augmentation_model.normalize_enabled = self.checkBox_normalize_enabled.isChecked()
        data_augmentation_model.normalize_mean1 = self.doubleSpinBox_mean1.value()
        data_augmentation_model.normalize_mean2 = self.doubleSpinBox_mean2.value()
        data_augmentation_model.normalize_mean3 = self.doubleSpinBox_mean3.value()
        data_augmentation_model.normalize_std1 = self.doubleSpinBox_std1.value()
        data_augmentation_model.normalize_std2 = self.doubleSpinBox_std2.value()
        data_augmentation_model.normalize_std3 = self.doubleSpinBox_std3.value()
        data_augmentation_model.center_crop_enabled = self.checkBox_center_crop_enabled.isChecked()
        data_augmentation_model.center_crop_image_size_single_enabled = self.checkBox_center_crop_image_size_single_enabled.isChecked()
        data_augmentation_model.center_crop_image_size_single = self.spinBox_center_crop_image_size_single.value()
        data_augmentation_model.center_crop_image_size_wh_enabled = self.checkBox_center_crop_image_size_wh_enabled.isChecked()
        data_augmentation_model.center_crop_image_width = self.spinBox_center_crop_image_width.value()
        data_augmentation_model.center_crop_image_height = self.spinBox_center_crop_image_height.value()
        data_augmentation_model.resize_enabled = self.checkBox_resize_enabled.isChecked()
        data_augmentation_model.resize_image_size_single_enabled = self.checkBox_resize_image_size_single_enabled.isChecked()
        data_augmentation_model.resize_image_size_single = self.spinBox_resize_image_size_single.value()
        data_augmentation_model.resize_image_size_wh_enabled = self.checkBox_resize_image_size_wh_enabled.isChecked()
        data_augmentation_model.resize_image_width = self.spinBox_resize_image_width.value()
        data_augmentation_model.resize_image_height = self.spinBox_resize_image_height.value()
        data_augmentation_model.random_horizontal_flip_enabled = self.checkBox_random_horizontal_flip_enabled.isChecked()
        data_augmentation_model.random_horizontal_flip_probability = self.doubleSpinBox_random_horizontal_flip_probability.value()
        data_augmentation_model.random_vertical_flip_enabled = self.checkBox_random_vertical_flip_enabled.isChecked()
        data_augmentation_model.random_vertical_flip_probability = self.doubleSpinBox_random_vertical_flip_probability.value()
        data_augmentation_model.color_jitter_enabled = self.checkBox_color_jitter_enabled.isChecked()
        data_augmentation_model.random_rotation_enabled = self.checkBox_random_rotation_enabled.isChecked()
        data_augmentation_model.random_rotation_angle = self.spinBox_random_rotation_angle.value()
        data_augmentation_model.random_resized_crop_enabled = self.checkBox_random_resized_crop_enabled.isChecked()
        data_augmentation_model.random_resized_crop_image_size_single_enabled = self.checkBox_random_resized_crop_image_size_single_enabled.isChecked()
        data_augmentation_model.random_resized_crop_image_size_single = self.spinBox_random_resized_crop_image_size_single.value()
        data_augmentation_model.random_resized_crop_image_size_wh_enabled = self.checkBox_random_resized_crop_image_size_wh_enabled.isChecked()
        data_augmentation_model.random_resized_crop_image_height = self.spinBox_random_resized_crop_image_height.value()
        data_augmentation_model.random_resized_crop_image_width = self.spinBox_random_resized_crop_image_width.value()
        data_augmentation_model.random_resized_crop_scale1 = self.doubleSpinBox_random_resized_crop_scale1.value()
        data_augmentation_model.random_resized_crop_scale2 = self.doubleSpinBox_random_resized_crop_scale2.value()
        data_augmentation_model.random_resized_crop_ratio1 = self.doubleSpinBox_random_resized_crop_ratio1.value()
        data_augmentation_model.random_resized_crop_ratio2 = self.doubleSpinBox_random_resized_crop_ratio2.value()
        data_augmentation_model.grayscale_enabled = self.checkBox_grayscale_enabled.isChecked()
        self.callback_fnc(data_augmentation_model)

    def set_checkBox_normalize_enabled(self):
        self.checkBox_normalize_enabled.setChecked(self.data_augmentation_model.normalize_enabled)

    def set_checkBox_center_crop_enabled(self):
        self.checkBox_center_crop_enabled.setChecked(self.data_augmentation_model.center_crop_enabled)

    def set_checkBox_center_crop_image_size_single_enabled(self):
        self.checkBox_center_crop_image_size_single_enabled.setChecked(
            self.data_augmentation_model.center_crop_image_size_single_enabled)

    def set_checkBox_center_crop_image_size_wh_enabled(self):
        self.checkBox_center_crop_image_size_wh_enabled.setChecked(
            self.data_augmentation_model.center_crop_image_size_wh_enabled)

    def set_checkBox_resize_enabled(self):
        self.checkBox_resize_enabled.setChecked(self.data_augmentation_model.resize_enabled)

    def set_checkBox_resize_image_size_single_enabled(self):
        self.checkBox_resize_image_size_single_enabled.setChecked(
            self.data_augmentation_model.resize_image_size_single_enabled)

    def set_checkBox_resize_image_size_wh_enabled(self):
        self.checkBox_resize_image_size_wh_enabled.setChecked(
            self.data_augmentation_model.resize_image_size_wh_enabled)

    def set_checkBox_random_horizontal_flip_enabled(self):
        self.checkBox_random_horizontal_flip_enabled.setChecked(
            self.data_augmentation_model.random_horizontal_flip_enabled)

    def set_checkBox_random_vertical_flip_enabled(self):
        self.checkBox_random_vertical_flip_enabled.setChecked(
            self.data_augmentation_model.random_vertical_flip_enabled)

    def set_checkBox_color_jitter_enabled(self):
        self.checkBox_color_jitter_enabled.setChecked(self.data_augmentation_model.color_jitter_enabled)

    def set_checkBox_random_rotation_enabled(self):
        self.checkBox_random_rotation_enabled.setChecked(self.data_augmentation_model.random_rotation_enabled)

    def set_checkBox_random_resized_crop_enabled(self):
        self.checkBox_random_resized_crop_enabled.setChecked(
            self.data_augmentation_model.random_resized_crop_enabled)

    def set_checkBox_random_resized_crop_image_size_single_enabled(self):
        self.checkBox_random_resized_crop_image_size_single_enabled.setChecked(
            self.data_augmentation_model.random_resized_crop_image_size_single_enabled)

    def set_checkBox_random_resized_crop_image_size_wh_enabled(self):
        self.checkBox_random_resized_crop_image_size_wh_enabled.setChecked(
            self.data_augmentation_model.random_resized_crop_image_size_wh_enabled)

    def set_checkBox_grayscale_enabled(self):
        self.checkBox_grayscale_enabled.setChecked(self.data_augmentation_model.grayscale_enabled)

    def set_doubleSpinBox_mean1(self):
        self.doubleSpinBox_mean1.setValue(self.data_augmentation_model.normalize_mean1)

    def set_doubleSpinBox_mean2(self):
        self.doubleSpinBox_mean2.setValue(self.data_augmentation_model.normalize_mean2)

    def set_doubleSpinBox_mean3(self):
        self.doubleSpinBox_mean3.setValue(self.data_augmentation_model.normalize_mean3)

    def set_doubleSpinBox_std1(self):
        self.doubleSpinBox_std1.setValue(self.data_augmentation_model.normalize_std1)

    def set_doubleSpinBox_std2(self):
        self.doubleSpinBox_std2.setValue(self.data_augmentation_model.normalize_std2)

    def set_doubleSpinBox_std3(self):
        self.doubleSpinBox_std3.setValue(self.data_augmentation_model.normalize_std3)

    def set_doubleSpinBox_random_resized_crop_scale1(self):
        self.doubleSpinBox_random_resized_crop_scale1.setValue(
            self.data_augmentation_model.random_resized_crop_scale1)

    def set_doubleSpinBox_random_resized_crop_scale2(self):
        self.doubleSpinBox_random_resized_crop_scale2.setValue(
            self.data_augmentation_model.random_resized_crop_scale2)

    def set_doubleSpinBox_random_resized_crop_ratio1(self):
        self.doubleSpinBox_random_resized_crop_ratio1.setValue(
            self.data_augmentation_model.random_resized_crop_ratio1)

    def set_doubleSpinBox_random_resized_crop_ratio2(self):
        self.doubleSpinBox_random_resized_crop_ratio2.setValue(
            self.data_augmentation_model.random_resized_crop_ratio2)

    def set_doubleSpinBox_random_horizontal_flip_probability(self):
        self.doubleSpinBox_random_horizontal_flip_probability.setValue(
            self.data_augmentation_model.random_horizontal_flip_probability)

    def set_doubleSpinBox_random_vertical_flip_probability(self):
        self.doubleSpinBox_random_vertical_flip_probability.setValue(
            self.data_augmentation_model.random_vertical_flip_probability)

    def set_spinBox_center_crop_image_size_single(self):
        self.spinBox_center_crop_image_size_single.setMaximum(SPINBOX_MAX_VALUE)
        self.spinBox_center_crop_image_size_single.setValue(
            self.data_augmentation_model.center_crop_image_size_single)

    def set_spinBox_center_crop_image_width(self):
        self.spinBox_center_crop_image_width.setMaximum(SPINBOX_MAX_VALUE)
        self.spinBox_center_crop_image_width.setValue(self.data_augmentation_model.center_crop_image_width)

    def set_spinBox_center_crop_image_height(self):
        self.spinBox_center_crop_image_height.setMaximum(SPINBOX_MAX_VALUE)
        self.spinBox_center_crop_image_height.setValue(self.data_augmentation_model.center_crop_image_height)

    def set_spinBox_resize_image_size_single(self):
        self.spinBox_resize_image_size_single.setMaximum(SPINBOX_MAX_VALUE)
        self.spinBox_resize_image_size_single.setValue(self.data_augmentation_model.resize_image_size_single)

    def set_spinBox_resize_image_width(self):
        self.spinBox_resize_image_width.setMaximum(SPINBOX_MAX_VALUE)
        self.spinBox_resize_image_width.setValue(self.data_augmentation_model.resize_image_width)

    def set_spinBox_resize_image_height(self):
        self.spinBox_resize_image_height.setMaximum(SPINBOX_MAX_VALUE)
        self.spinBox_resize_image_height.setValue(self.data_augmentation_model.resize_image_height)

    def set_spinBox_random_rotation_angle(self):
        self.spinBox_random_rotation_angle.setMaximum(SPINBOX_MAX_VALUE)
        self.spinBox_random_rotation_angle.setValue(self.data_augmentation_model.random_rotation_angle)

    def set_spinBox_random_resized_crop_image_size_single(self):
        self.spinBox_random_resized_crop_image_size_single.setMaximum(SPINBOX_MAX_VALUE)
        self.spinBox_random_resized_crop_image_size_single.setValue(
            self.data_augmentation_model.random_resized_crop_image_size_single)

    def set_spinBox_random_resized_crop_image_width(self):
        self.spinBox_random_resized_crop_image_width.setMaximum(SPINBOX_MAX_VALUE)
        self.spinBox_random_resized_crop_image_width.setValue(
            self.data_augmentation_model.random_resized_crop_image_width)

    def set_spinBox_random_resized_crop_image_height(self):
        self.spinBox_random_resized_crop_image_height.setMaximum(SPINBOX_MAX_VALUE)
        self.spinBox_random_resized_crop_image_height.setValue(
            self.data_augmentation_model.random_resized_crop_image_height)
