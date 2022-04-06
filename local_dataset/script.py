import os
import re


def get_face2comic_image_paths():
    rootdir = '/media/x/disk/dataset/comic_faces/comics'
    list = os.listdir(rootdir)
    with open('../data/face2comic/train_comics.txt', 'w') as f:
        f.truncate(0)
        for i in range(0, int(len(list) * 0.8)):
            path = os.path.join(rootdir, f'{i}.jpg')
            f.write(path)
            f.write('\n')

    with open('../data/face2comic/test_comics.txt', 'w') as f:
        f.truncate(0)
        for i in range(int(len(list) * 0.8), len(list)):
            path = os.path.join(rootdir, f'{i}.jpg')
            f.write(path)
            f.write('\n')

    rootdir = '/media/x/disk/dataset/comic_faces/face'
    list = os.listdir(rootdir)
    with open('../data/face2comic/train_face.txt', 'w') as f:
        f.truncate(0)
        for i in range(0, int(len(list) * 0.8)):
            path = os.path.join(rootdir, f'{i}.jpg')
            f.write(path)
            f.write('\n')

    with open('../data/face2comic/test_face.txt', 'w') as f:
        f.truncate(0)
        for i in range(int(len(list) * 0.8), len(list)):
            path = os.path.join(rootdir, f'{i}.jpg')
            f.write(path)
            f.write('\n')


def get_cityscape_pairs_paths():
    rootdir = '/media/x/disk/dataset/cityscape_pairs/train'
    list = os.listdir(rootdir)
    with open('../data/cityscape_pairs/train.txt', 'w') as f:
        f.truncate(0)
        for i in range(0, len(list)):
            path = os.path.join(rootdir, f'{i + 1}.jpg')
            f.write(path)
            f.write('\n')

    rootdir = '/media/x/disk/dataset/cityscape_pairs/test'
    list = os.listdir(rootdir)
    with open('../data/cityscape_pairs/test.txt', 'w') as f:
        f.truncate(0)
        for i in range(0, len(list)):
            path = os.path.join(rootdir, f'{i + 1}.jpg')
            f.write(path)
            f.write('\n')


def get_butterfly_paths():
    rootdir = '/media/x/disk/dataset/butterfly/images'
    list = os.listdir(rootdir)
    list.sort()
    with open('../data/butterfly/train_img.txt', 'w') as f:
        f.truncate(0)
        for i in range(0, int(len(list) * 0.8)):
            path = os.path.join(rootdir, list[i])
            f.write(path)
            f.write('\n')

    with open('../data/butterfly/test_img.txt', 'w') as f:
        f.truncate(0)
        for i in range(int(len(list) * 0.8), len(list)):
            path = os.path.join(rootdir, list[i])
            f.write(path)
            f.write('\n')

    rootdir = '/media/x/disk/dataset/butterfly/segmentations'
    list = os.listdir(rootdir)
    list.sort()
    with open('../data/butterfly/train_seg.txt', 'w') as f:
        f.truncate(0)
        for i in range(0, int(len(list) * 0.8)):
            path = os.path.join(rootdir, list[i])
            f.write(path)
            f.write('\n')

    with open('../data/butterfly/test_seg.txt', 'w') as f:
        f.truncate(0)
        for i in range(int(len(list) * 0.8), len(list)):
            path = os.path.join(rootdir, list[i])
            f.write(path)
            f.write('\n')


def get_cityscape_paths(type='train'):
    img_txt = f'../data/cityscape/{type}_img.txt'
    seg_txt = f'../data/cityscape/{type}_seg.txt'
    img_rootdir = os.path.join('/media/x/disk/dataset/cityscape/IMAGE', type)
    seg_rootdir = os.path.join('/media/x/disk/dataset/cityscape/gtFine', type)

    with open(img_txt, 'w') as f:
        f.truncate(0)
        dirlist = os.listdir(img_rootdir)
        dirlist.sort()
        for i in range(0, len(dirlist)):
            dirpath = os.path.join(img_rootdir, dirlist[i])
            imagelist = os.listdir(dirpath)
            imagelist.sort()
            for j in range(len(imagelist)):
                imagepath = os.path.join(dirpath, imagelist[j])
                f.write(imagepath)
                f.write('\n')

    with open(seg_txt, 'w') as f:
        f.truncate(0)
        dirlist = os.listdir(seg_rootdir)
        dirlist.sort()
        for i in range(0, len(dirlist)):
            dirpath = os.path.join(seg_rootdir, dirlist[i])
            imagelist = os.listdir(dirpath)
            imagelist.sort()
            for j in range(len(imagelist)):
                matchobj = re.match(r'(.*)color(.*)', imagelist[j])
                if matchobj:
                    imagepath = os.path.join(dirpath, imagelist[j])
                    f.write(imagepath)
                    f.write('\n')


def get_fish_paths():
    train_img_txt = '../data/fish/train_img.txt'
    test_img_txt = '../data/fish/test_img.txt'
    train_seg_txt = '../data/fish/train_seg.txt'
    test_seg_txt = '../data/fish/test_seg.txt'
    rootdir = os.path.join(f'/media/x/disk/dataset/fish', 'train')

    with open(train_img_txt, 'w') as f_train_img, \
         open(test_img_txt, 'w') as f_test_img, \
         open(train_seg_txt, 'w') as f_train_seg, \
         open(test_seg_txt, 'w') as f_test_seg:
        f_train_img.truncate(0)
        f_test_img.truncate(0)
        f_train_seg.truncate(0)
        f_test_seg.truncate(0)
        dirlist = os.listdir(rootdir)
        dirlist.sort()
        for i in range(0, len(dirlist)):
            subdirpath = os.path.join(rootdir, dirlist[i])
            subdirlist = os.listdir(subdirpath)
            subdirlist.sort()
            for j in range(len(subdirlist)):
                subsubdirpath = os.path.join(subdirpath, subdirlist[j])
                imagelist = os.listdir(subsubdirpath)
                imagelist.sort()
                matchobj = re.match(r'(.*)GT', subdirlist[j])
                if matchobj:
                    for k in range(int(len(imagelist) * 0.9)):
                        imagepath = os.path.join(subsubdirpath, imagelist[k])
                        f_train_seg.write(imagepath)
                        f_train_seg.write('\n')
                    for k in range(int(len(imagelist) * 0.9), len(imagelist)):
                        imagepath = os.path.join(subsubdirpath, imagelist[k])
                        f_test_seg.write(imagepath)
                        f_test_seg.write('\n')
                else:
                    for k in range(int(len(imagelist) * 0.9)):
                        imagepath = os.path.join(subsubdirpath, imagelist[k])
                        f_train_img.write(imagepath)
                        f_train_img.write('\n')
                    for k in range(int(len(imagelist) * 0.9), len(imagelist)):
                        imagepath = os.path.join(subsubdirpath, imagelist[k])
                        f_test_img.write(imagepath)
                        f_test_img.write('\n')