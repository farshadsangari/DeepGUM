from torchvision import transforms
from PIL import Image

import re
import torch.utils.data as data
from .prepare_data import split_dirs


class CACD_Dataloader(data.Dataset):
    def __init__(self, dirs, mean, std):
        self.images = dirs
        self.mean, self.std = mean, std
        self.ages = [int(re.search("(\d*)_.*", x).group(1)) for x in dirs]
        self.names = [
            re.search("\d*_(.*)_\d*.jpg", x).group(1).replace("_", " ") for x in dirs
        ]
        self.transformers = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __getitem__(self, index):
        image = self.images[index]
        age = self.ages[index]
        name = self.names[index]

        # reading images with PIL
        image = Image.open(image)
        image = self.transformers(image)
        age = (age - self.mean) / self.std

        return image, age, name

    def __len__(self):
        return len(self.images)


def dataloaders(
    all_dirs_path,
    num_train_data,
    num_test_data,
    num_val_data,
    batch_size,
    shuffle,
    num_workers,
):

    train_dirs, test_dirs, val_dirs = split_dirs(
        all_dirs_path, num_train_data, num_test_data, num_val_data, batch_size
    )
    cacd_train_data = CACD_Dataloader(train_dirs)
    cacd_trainLoader = data.DataLoader(
        cacd_train_data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )

    cacd_test_data = CACD_Dataloader(test_dirs)
    cacd_testLoader = data.DataLoader(
        cacd_test_data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )

    cacd_val_data = CACD_Dataloader(val_dirs)
    cacd_valLoader = data.DataLoader(
        cacd_val_data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )

    return cacd_trainLoader, cacd_testLoader, cacd_valLoader
