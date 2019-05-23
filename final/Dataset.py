from enums.PyTorchModelsEnum import PyTorchModelsEnum

from final.GlobalSettings import GlobalSettings
from torch.utils.data import DataLoader


from torchvision import datasets, transforms


class Dataset:
    def __init__(self):
        self.data = None
        self.data_loaders = None

        self.run()

    def run(self):
        self.data, self.data_loaders = self.create_dataset()

    def create_dataset(self):
        image_transforms = self.data_augmentation()

        data = {
            GlobalSettings.TRAIN_DIRNAME:
                datasets.ImageFolder(
                    root=GlobalSettings.TRAIN_DIR,
                    transform=image_transforms[GlobalSettings.TRAIN_DIRNAME]
                ),
            GlobalSettings.VALID_DIRNAME:
                datasets.ImageFolder(
                    root=GlobalSettings.VALID_DIR, transform=image_transforms[GlobalSettings.VALID_DIRNAME]
                ),
            GlobalSettings.TEST_DIRNAME:
                datasets.ImageFolder(
                    root=GlobalSettings.TEST_DIR, transform=image_transforms[GlobalSettings.TEST_DIRNAME]
                )
        }

        data_loaders = {
            GlobalSettings.TRAIN_DIRNAME: DataLoader(
                data[GlobalSettings.TRAIN_DIRNAME],
                batch_size=GlobalSettings.BATCH_SIZE,
                shuffle=True
            ),
            GlobalSettings.VALID_DIRNAME: DataLoader(
                data[GlobalSettings.VALID_DIRNAME],
                batch_size=GlobalSettings.BATCH_SIZE,
                shuffle=True
            ),
            GlobalSettings.TEST_DIRNAME: DataLoader(
                data[GlobalSettings.TEST_DIRNAME],
                batch_size=GlobalSettings.BATCH_SIZE,
                shuffle=True
            )
        }

        return data, data_loaders

    @staticmethod
    def data_augmentation():
        if GlobalSettings.CNN_MODEL == PyTorchModelsEnum.INCEPTION_V3:
            GlobalSettings.IMG_SIZE = 299

        return {
            GlobalSettings.TRAIN_DIRNAME:
                transforms.Compose([
                    transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
                    transforms.RandomRotation(degrees=15),
                    transforms.ColorJitter(),
                    transforms.RandomHorizontalFlip(),
                    transforms.CenterCrop(size=GlobalSettings.IMG_SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])
                ]),
            GlobalSettings.VALID_DIRNAME:
                transforms.Compose([
                    transforms.Resize(size=256),
                    transforms.CenterCrop(size=GlobalSettings.IMG_SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]),
            GlobalSettings.TEST_DIRNAME:
                transforms.Compose([
                    transforms.Resize(size=256),
                    transforms.CenterCrop(size=GlobalSettings.IMG_SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
        }
