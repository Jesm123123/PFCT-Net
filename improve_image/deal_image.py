import os
import cv2
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class MedicalImageDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_size, output_channels, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.output_channels = output_channels
        self.images = os.listdir(image_dir)
        self.size = image_size

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_name = self.images[idx]
        image_path = os.path.join(self.image_dir, image_name)

        image = cv2.imread(image_path)
        if self.output_channels == 1:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif self.output_channels == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB


        image = cv2.resize(image, self.size)
        if self.mask_dir:
            mask_name = image_name
            mask_path = os.path.join(self.mask_dir, mask_name)
            mask = Image.open(mask_path).convert("L")
            mask = mask.resize(self.size)
        else:
            mask = Image.new("L", self.size, 0)

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        else:
            if self.output_channels == 3:
                image = torch.tensor(np.array(image) / 255., dtype=torch.float32).permute(2, 0, 1)
            elif self.output_channels == 1:
                image = torch.tensor(np.array(image) / 255., dtype=torch.float32).unsqueeze(0)
            mask = torch.tensor(np.array(mask) / 255., dtype=torch.float32).unsqueeze(0)

        return image, mask


# image_path = "./../data/CVC-ColonDB-20240120T060139Z-001/CVC-ColonDB/images"
# mask_path = "./../data/CVC-ColonDB-20240120T060139Z-001/CVC-ColonDB/masks"
# MA_color = MedicalImageDataset(image_path, mask_path, image_size=(256, 256), output_channels=1)
#
# # Example usage
# image_color, mask_color = MA_color[0]  # Outputting color image
# print(image_color.shape, mask_color.shape)
