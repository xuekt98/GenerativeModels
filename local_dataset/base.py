import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image


class ImagePathDataset(Dataset):
    def __init__(self, image_paths, image_size=(256,256), to_rgb=False):
        self.image_size = image_size
        self.image_paths = image_paths
        self._length = len(image_paths)
        self.to_rgb = to_rgb
        self.transform = transforms.Compose([
                            transforms.Resize(self.image_size),
                            transforms.ToTensor()
                        ])

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        image=None
        try:
            image = Image.open(img_path)
        except BaseException as e:
            print(img_path)

        if self.to_rgb:
            image = image.convert('RGB')
        image = self.transform(image)
        return image

