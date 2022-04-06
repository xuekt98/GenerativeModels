from torch.utils.data import Dataset
from local_dataset.base import ImagePathDataset


class CityScapeDataset(Dataset):
    def __init__(self, type='train', image_size=(256,256)):
        super().__init__()
        self.image_size = image_size
        with open(f'./data/cityscape/{type}_img.txt', "r") as f:
            image_paths = f.read().splitlines()
            self.img = ImagePathDataset(image_paths, self.image_size)

        with open(f'./data/cityscape/{type}_seg.txt', "r") as f:
            image_paths = f.read().splitlines()
            self.seg = ImagePathDataset(image_paths, self.image_size, to_rgb=True)

    def __len__(self):
        return len(self.img)

    def __getitem__(self, i):
        return self.img[i], self.seg[i]