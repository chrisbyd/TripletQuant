from utils.tools import *
from models import Transformer as CrossVisionTransformer
import os
import os.path as osp
import torch
import torch.optim as optim
import time
import torch.nn as nn
from datasets import get_data
import numpy as np
import random
from models import Model, AlexNet
from validate import pq_validate
from tqdm import tqdm
# from utils.fastap_loss import FastAPLoss
from utils.adapt_fastap_loss import FastAPLoss 
from utils.triplet_loss import batch_all_triplet_loss
from utils.loss import SupConLoss, N_PQ_loss, CLS_loss, raw_hash_loss, CLS_bce_loss, TripletLoss
from utils.functions import intra_norm, my_soft_assignment
from networks import GPQSoftMaxNet


def train(config):
    
    num_class = config['n_class']
    len_code = config['len_code']
    intn_word = config['intn_word']
    n_book = config['n_book']
    alpha = config['alpha']
    beta = config['beta']
    lam_2 = config['lam_2']
    train_loader, test_loader, dataset_loader, num_train, num_test, num_dataset = get_data(config)
    Ap_loss = FastAPLoss()
    config["num_train"] = num_train
    feat_dim = config['len_code'] * config['n_book']
    model = Model(config= config)
    model.initialize()
    net = model.cuda()

    gpq = GPQSoftMaxNet(True, n_classes=num_class, n_book=n_book, intn_word=intn_word, len_code=len_code)

    optimizer = optim.RMSprop(net.parameters(),lr= 1e-5, weight_decay =1e-5)
    zoptimizer = optim.RMSprop([gpq.Z], lr =  1e-5, weight_decay = 1e-5)
    poptimizer = optim.RMSprop([gpq.Prototypes], lr = 1e-5, weight_decay = 1e-5 )
    foptimizer = optim.RMSprop(gpq.fake_linear.parameters(), lr = 1e-5, weight_decay = 1e-5)
    Best_mAP = 0
    print("The experiment name is", config['exp_name'])

    for epoch in tqdm(range(config["epoch"])):

        current_time = time.strftime('%H:%M:%S', time.localtime(time.time()))

        print("%s[%2d/%2d][%s] bit:%d, dataset:%s, training...." % (
            config["exp_name"], epoch + 1, config["epoch"], current_time, n_book * config['bn_word'], config["dataset_name"]), end="")

        net.train()
        old_Z = None
        train_loss = 0
        old_P = None
        for image, label, ind in tqdm(train_loader):
            # x1 = image[0].cuda()
            # x2 = image[1].cuda()
            # image = torch.cat([x1,x2],dim=0)
            
            image = image.cuda()

            # label = torch.cat([label,label], dim = 0)
            label = label.type(torch.FloatTensor).cuda()
            # print('label', label.shape)
            labelOneHot = label.to(torch.float)
            # print('onehot', labelOneHot.shape)
        
            foptimizer.zero_grad()
            optimizer.zero_grad()
            zoptimizer.zero_grad()
            poptimizer.zero_grad()
            # calculate original feature
            u = net(image)
           
            # norm feature and protoype
            # u = intra_norm(u, len_code, n_book)
            
   
            # Z = intra_norm(gpq.Z, len_code, n_book)
            Z = gpq.Z
          
            # #print('@@@@', gpq.Z)
            descriptor_S = my_soft_assignment(Z, u, len_code, alpha, device='cuda')
            descriptor_S = torch.nn.functional.normalize(descriptor_S, dim=-1)
            # logits_fake = gpq.fake_C(descriptor_S) 

            # con_loss_asym = ContrastLoss(u,label,descriptor_S,label)
         #   con_loss_feature = ContrastLoss(u,label,u,label)
            # con_loss_descriptor = ContrastLoss(descriptor_S,label,descriptor_S,label)
           # hybrid_con_loss = con_loss_asym 
            
          #  ap_loss = Ap_loss.compute_loss(u,descriptor_S,label)
        
           # loss = hash_loss + lam_2 * clss_loss + con_loss_asym + con_loss_descriptor
          #  loss =   ap_loss
            loss = batch_all_triplet_loss(label,label,u,descriptor_S, margin=0.6 )
            train_loss += loss.item()

            #print('clss_loss', clss_loss.item(), hash_loss.item())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()
            zoptimizer.step()
            poptimizer.step()
            foptimizer.step()
        # if "-A" in config["info"]:
        #     criterion.update_target_vectors()

        train_loss = train_loss / len(train_loader)

        print("\b\b\b\b\b\b\b loss:%.3f" % (train_loss))
        save_path = osp.join(config['checkpoint_dir'], config['exp_name'])
        save_z_path = osp.join(save_path, 'gpq')
        if not osp.exists(save_path):
            os.makedirs(save_path)
        if not osp.exists(save_z_path):
            os.makedirs(save_z_path)
        print("the eval interval is", config["eval_interval"] )
          
        if (epoch + 1) % config["eval_interval"] == 0:
            # save
            print("Epoch {}, start validating...".format(epoch))
            
            # print("calculating test binary code......")
            with torch.no_grad():
                mAP = pq_validate(config,gpq.Z,epoch_num=epoch, net=net)
                
       #     mAP = validate(config, f_sz, epoch_num=epoch, best_map=Best_mAP, net=net, isHamming=False)
            

            if mAP > Best_mAP:
                Best_mAP = mAP
                torch.save(net.state_dict(),
                       os.path.join(save_path, config["dataset_name"] + "_" + str(mAP) + '_' + str(n_book * config['bn_word']) + '_' +"model.pt"))
            gpq.save(os.path.join(save_z_path,config['dataset_name'] + "_"+ str(mAP) +'_' + str(n_book * config['bn_word']) +'_'+'z.pt'))
            print("%s epoch:%d, bit:%d, dataset:%s, MAP:%.3f, Best MAP: %.3f" % (
                config["exp_name"], epoch + 1, n_book * config['bn_word'], config["dataset_name"], mAP, Best_mAP))
            print(config)
