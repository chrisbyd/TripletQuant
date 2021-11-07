from utils.tools import *
from networks import *
from models import Transformer as CrossVisionTransformer
import os
import os.path as osp
import torch
import torch.optim as optim
import time
import torch.nn as nn
import numpy as np
import random
from models import Model
from validate import pq_validate
from tqdm import tqdm
from utils.loss import SupConLoss, N_PQ_loss, CLS_loss, raw_hash_loss
from utils.functions import intra_norm, my_soft_assignment



def evaluate(config):
    num_class = config['n_class']
    len_code = config['len_code']
    intn_word = config['intn_word']
    n_book = config['n_book']
    bit_length = config['n_book']* config['bn_word']

    
    # defining the dir for all the models 
    save_dir = osp.join(config['checkpoint_dir'], config['exp_name'])
    gpq_save_dir =  osp.join(save_dir, 'gpq')
    code_save_dir = osp.join(save_dir, 'codes')

    prefix_save_name = config['dataset_name'] + '_' + str(config['n_book']* config['bn_word']) + '_'
    model_save_path = osp.join(save_dir, prefix_save_name + 'model.pt')
    gpq_save_path = osp.join(gpq_save_dir, prefix_save_name + 'z.pt')
    code_save_path = osp.join(code_save_dir, prefix_save_name + 'code.npy')
    # load 
    if config['use_p_code']:
        pre_computed_codes = np.load(code_save_path,allow_pickle=True).item()
        mAP = pq_validate(config, if_save_code= 0, precomputed_codes= pre_computed_codes)
        print("Model name: {}, bit number:  {}, dataset name: {}, the mAP is {}".format(config['exp_name'], 
                             str(bit_length), config['dataset_name'], mAP ))
    else:
        # defining the neural network model
        model = Model(config= config)
        gpq = GPQSoftMaxNet(True, n_classes=num_class, n_book=n_book, intn_word=intn_word, len_code=len_code)

        model.load_state_dict(torch.load(model_save_path))
        model = model.cuda()
        Z = gpq.load(gpq_save_path)
        mAP = pq_validate(config, z = Z, net = model, if_save_code= 0, precomputed_codes= None)
        print("Model name: {}, bit number:  {}, dataset name: {}, the mAP is {}".format(config['exp_name'], 
                             str(bit_length), config['dataset_name'], mAP ))

