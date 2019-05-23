from enums.PyTorchModelsEnum import PyTorchModelsEnum


class GlobalSettings:
    BATCH_SIZE = 128

    DATASET_DIR = './data/caltech'
    TRAIN_DIRNAME = 'train'
    VALID_DIRNAME = 'valid'
    TEST_DIRNAME = 'test'

    TRAIN_DIR = DATASET_DIR + '/' + TRAIN_DIRNAME
    VALID_DIR = DATASET_DIR + '/' + VALID_DIRNAME
    TEST_DIR = DATASET_DIR + '/' + TEST_DIRNAME

    FEATURE_EXTRACT = False
    CNN_MODEL = PyTorchModelsEnum.RESNET152
    CNN_USE_SIMPLE_CLASSIFIER = True
    NUM_EPOCHS = 5

    IMG_SIZE = 224

    SAVE_FOLDER = './saveFolder'

    NUM_CLASSES = 3