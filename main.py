from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import fire
from train import train
import os.path as osp
from configs.config import ex
import logging
import copy
from test import evaluate
torch.multiprocessing.set_sharing_strategy('file_system')




#def train(config, train_loader,model):

@ex.automain
def main(_config):
         cfg  = copy.deepcopy(_config)
         # get the model to train
         if cfg['mode'] == "train":
             train(config= cfg)
         elif cfg['mode'] == 'test':
             evaluate(config= cfg)
         else:
             raise NotImplementedError("")         

         
