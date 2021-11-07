from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import torch
import numpy as np



def image_transform(resize_size, crop_size, data_set):
    if data_set == "train":
        step = [transforms.RandomHorizontalFlip(), transforms.RandomCrop(crop_size)]
    else:
        step = [transforms.CenterCrop(crop_size)]
    return transforms.Compose([transforms.Resize(resize_size)]
                              + step +
                              [transforms.ToTensor(),
                               transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])
                               ])


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std= [0.229, 0.224, 0.225])

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(size=(224,224), scale=(0.2, 1.)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply([
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
    ], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    normalize,
])

class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]



