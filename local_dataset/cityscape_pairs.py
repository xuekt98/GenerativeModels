import pdb

from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image


class CityScapePairsDataset(Dataset):
    def __init__(self, type='train', image_size=(256,256)):
        super().__init__()
        if type == 'train':
            with open('./data/cityscape_pairs/train.txt', "r") as f:
                self.image_paths = f.read().splitlines()
        else:
            with open('./data/cityscape_pairs/test.txt', "r") as f:
                self.image_paths = f.read().splitlines()
        self.image_size = image_size
        self._length = len(self.image_paths)
        self.transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor()
        ])

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        image = Image.open(img_path)
        city = image.crop((0, 0, 256, 256))
        segm = image.crop((256, 0, 512, 256))
        city = self.transform(city)
        segm = self.transform(segm)
        return city, segm