import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from albumentations import Compose, HorizontalFlip
import numpy as np

class Cdata(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

        self.images = os.listdir(self.image_dir)
        self.masks = os.listdir(self.mask_dir)

    def __len__(self):
        return len(self.images)  # 返回数据集的大小

    def __getitem__(self, index):
        img_name = self.images[index]
        mask_name = self.masks[index]

        image_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            augmented = self.transform(image=np.array(image), mask=np.array(mask))
            image = augmented['image']
            mask = augmented['mask']

        return image, mask