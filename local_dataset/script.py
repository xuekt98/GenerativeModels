from torch.utils.data import Dataset, DataLoader
import os
import shutil
import pdb
from tqdm import tqdm
from PIL import Image


def get_face2comic_image_paths():
    rootdir = '/media/x/disk/local_dataset/comic_faces/comics'
    list = os.listdir(rootdir)
    with open('../data/face2comic/train_comics.txt', 'w') as f:
        for i in range(0, int(len(list) * 0.8)):
            path = os.path.join(rootdir, f'{i}.jpg')
            f.write(path)
            f.write('\n')

    with open('../data/face2comic/test_comics.txt', 'w') as f:
        for i in range(int(len(list) * 0.8), len(list)):
            path = os.path.join(rootdir, f'{i}.jpg')
            f.write(path)
            f.write('\n')

    rootdir = '/media/x/disk/local_dataset/comic_faces/face'
    list = os.listdir(rootdir)
    with open('../data/face2comic/train_face.txt', 'w') as f:
        for i in range(0, int(len(list) * 0.8)):
            path = os.path.join(rootdir, f'{i}.jpg')
            f.write(path)
            f.write('\n')

    with open('../data/face2comic/test_face.txt', 'w') as f:
        for i in range(int(len(list) * 0.8), len(list)):
            path = os.path.join(rootdir, f'{i}.jpg')
            f.write(path)
            f.write('\n')


get_face2comic_image_paths()