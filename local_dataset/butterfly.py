from torch.utils.data import Dataset
from local_dataset.base import ImagePathDataset


class ButterflyDataset(Dataset):
    def __init__(self, type='train', image_size=(256,256)):
        super().__init__()
        self.image_size = image_size
        if type == 'train':
            with open('./data/butterfly/train_img.txt', "r") as f:
                image_paths = f.read().splitlines()
                self.imgs = ImagePathDataset(image_paths, self.image_size)
            with open('./data/butterfly/train_seg.txt', "r") as f:
                image_paths = f.read().splitlines()
                self.segs = ImagePathDataset(image_paths, self.image_size, to_rgb=True)
        else:
            with open('./data/butterfly/test_img.txt', "r") as f:
                image_paths = f.read().splitlines()
                self.imgs = ImagePathDataset(image_paths, self.image_size)
            with open('./data/butterfly/test_seg.txt', "r") as f:
                image_paths = f.read().splitlines()
                self.segs = ImagePathDataset(image_paths, self.image_size, to_rgb=True)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        return self.imgs[i], self.segs[i]