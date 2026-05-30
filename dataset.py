from torch.utils.data import Dataset
from PIL import Image

import torchvision.transforms as transforms
import os

class SRDataset(Dataset):

    def __init__(self, hr_dir):

        self.hr_dir = hr_dir

        self.images = os.listdir(hr_dir)

        self.to_tensor = transforms.ToTensor()

        self.hr_crop = transforms.RandomCrop((256,256))

        self.augment = transforms.Compose([

            transforms.RandomHorizontalFlip(),

            transforms.RandomRotation(10),

        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        image_name = self.images[idx]

        hr_path = os.path.join(
            self.hr_dir,
            image_name
        )

        hr_image = Image.open(
            hr_path
        ).convert("RGB")

        hr_image = self.hr_crop(hr_image)

        hr_image = self.augment(hr_image)

        lr_image = hr_image.resize(
            (64,64),
            Image.BICUBIC
        )

        hr_tensor = self.to_tensor(hr_image)

        lr_tensor = self.to_tensor(lr_image)

        return lr_tensor, hr_tensor