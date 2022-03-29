from torch.utils.data import Dataset
from local_dataset.base import ImagePathDataset


class Face2ComicDataset(Dataset):
    def __init__(self, type='train', image_size=(256,256)):
        super().__init__()
        self.image_size = image_size
        if type == 'train':
            with open('./data/face2comic/train_comics.txt', "r") as f:
                image_paths = f.read().splitlines()
                self.comics = ImagePathDataset(image_paths, self.image_size)
            with open('./data/face2comic/train_face.txt', "r") as f:
                image_paths = f.read().splitlines()
                self.face = ImagePathDataset(image_paths, self.image_size)
        else:
            with open('./data/face2comic/test_comics.txt', "r") as f:
                image_paths = f.read().splitlines()
                self.comics = ImagePathDataset(image_paths, self.image_size)
            with open('./data/face2comic/test_face.txt', "r") as f:
                image_paths = f.read().splitlines()
                self.face = ImagePathDataset(image_paths, self.image_size)

    def __len__(self):
        return len(self.comics)

    def __getitem__(self, i):
        return self.comics[i], self.face[i]