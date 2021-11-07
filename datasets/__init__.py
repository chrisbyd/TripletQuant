from .base import TwoCropTransform, image_transform, train_transform
from torch.utils.data import DataLoader
import torch.utils.data as util_data
import os.path as osp
import numpy as np
from PIL import Image
from tqdm import tqdm




def get_data(config):

    dsets = {}
    dset_loaders = {}
    data_root = osp.join(config['data_root'], config['dataset_name'])

    for data_set in ["train", "test", "database"]:
        if data_set == 'train': 
            dsets[data_set] = ImageList(data_root,
                                        open(osp.join(data_root, data_set + '.txt')).readlines(),
                                        transform=image_transform(config["resize_size"], config["crop_size"], data_set))
            print(data_set, len(dsets[data_set]))
            dset_loaders[data_set] = util_data.DataLoader(dsets[data_set],
                                                      batch_size=config['batch_size'],
                                                      shuffle=True, num_workers=4)
        else:
            dsets[data_set] = ImageList(data_root,
                                        open(osp.join(data_root, data_set + '.txt')).readlines(),
                                        transform=image_transform(config["resize_size"], config["crop_size"], data_set))
            print(data_set, len(dsets[data_set]))
            dset_loaders[data_set] = util_data.DataLoader(dsets[data_set],
                                                      batch_size=config['test_batch_size'],
                                                      shuffle=True, num_workers=4)

    return dset_loaders["train"], dset_loaders["test"], dset_loaders["database"], \
           len(dsets["train"]), len(dsets["test"]), len(dsets["database"])




class ImageList(object):

    def __init__(self, data_path, image_list, transform):
        self.imgs = [(data_path +'/'+ val.split()[0], np.array([int(la) for la in val.split()[1:]])) for val in image_list]
        self.transform = transform

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        return img, target, index

    def __len__(self):
        return len(self.imgs)


if __name__ == '__main__': 
    data_list_path = '../data/coco2014/test.txt'
    data_root = '../data/coco2014/images/'
    transforms = get_transforms(256,224)
    cfg = {'pretrained_dir': '../pretrained/'}
    dataset = ImageTextDataset(cfg, data_list_path, transforms['train'])
    dataset[1]
