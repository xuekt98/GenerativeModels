import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image


class ImagePathDataset(Dataset):
    def __init__(self, image_paths, image_size=(256,256)):
        self.image_size = image_size
        self.image_paths = image_paths
        self._length = len(image_paths)
        self.transform = transforms.Compose([
                            transforms.Resize(self.image_size),
                            transforms.ToTensor()
                        ])

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        flip = False
        if index >= self._length:
            index = index - self._length
            flip = True

        img_path = self.image_paths[index]
        image = Image.open(img_path)
        image = self.transform(image)
        return image
