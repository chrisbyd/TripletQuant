import torch
import os 
from utils.tools import compute_result2
from utils.metric import MAPs
from utils.tools import CalcTopMap
from datasets import get_data
from utils.log_to_txt import results_to_txt
import numpy as np
from networks import GPQSoftMaxNet
from utils.functions import norm_gallery
from utils.functions import intra_norm


def pq_validate(config, z = None , epoch_num = 0 , net =None, if_save_code =1, precomputed_codes = None):
  
    n_book = config['n_book']
    len_code = config['len_code']
    train_loader, test_loader, dataset_loader, num_train, num_test, num_dataset = get_data(config)
    if precomputed_codes is None:
        if net is None:
            raise NotImplementedError("The model should be initialized! Should not be None")

        z = z.cpu()
       # z = intra_norm(z, config['len_code'], config['n_book'])
        net.eval()
        
        print("The number of gallery points is",num_dataset)
        query_codes, query_labels = (test_loader, net)
      #  query_codes = intra_norm(query_codes, config['len_code'] ,config['n_book'])
        # print("calculating dataset binary code.......")\
        gallery_codes, gallery_labels = compute_result2(dataset_loader, net)
        # gallery_codes = intra_norm(gallery_codes, config['len_code'] ,config['n_book'])
        gallery_codes = norm_gallery(z, gallery_codes, len_code, n_book)
        


    else:
        query_codes, query_labels, gallery_codes, gallery_labels = precomputed_codes['q_codes'], precomputed_codes['q_labels'], \
                                                                   precomputed_codes['g_codes'], precomputed_codes['g_labels']


        # print("calculating map.......")
    mAP, cum_prec, cum_recall = CalcTopMap(query_codes.detach().numpy(), gallery_codes.detach().numpy(), query_labels.detach().numpy(), gallery_labels.detach().numpy(),
                               config["topK"],distance_metric= config['dis_metric'] , config=config)

    metric = MAPs(config['topK'])
    # prec, recall, all_map = metric.get_precision_recall_by_Hamming_Radius_All(query_codes.numpy(),
    #                                                                           query_labels.numpy(),
    #                                                                           gallery_codes.numpy(),
    #                                                                           gallery_labels.numpy())
   # print(type(config['machine_name']), type(config['dataset_name']))
    file_name = str(config['machine_name']) +'_' +config['dataset_name']
    model_name = config['exp_name'] + '_' + str(n_book * config['bn_word'])+ '_' + str(epoch_num)
    index_range = num_dataset // 100
    index = [i * 100 - 1 for i in range(1, index_range+1)]

    max_index = max(index)
    overflow = num_dataset - index_range * 100
    index = index + [max_index + i  for i in range(1,overflow + 1)]

    c_prec = cum_prec[index].tolist()
    c_recall = cum_recall[index].tolist()


    results_to_txt([mAP], filename=file_name, model_name=model_name, sheet_name='map')
    results_to_txt(c_prec, filename=file_name, model_name=model_name, sheet_name='prec_cum')
    results_to_txt(c_recall, filename=file_name, model_name=model_name, sheet_name='recall_cum')
    # results_to_txt(prec.tolist(), filename=file_name, model_name=model_name, sheet_name='prec')
    # results_to_txt(recall.tolist(), filename=file_name, model_name=model_name, sheet_name='recall')

    mAP = int(mAP * 1000)/ 1000


    fname =  config['dataset_name'] + '_' + str(mAP) + '_'+ str(n_book * config['bn_word']) + '_' + 'code.npy'
    save_code_path = os.path.join(config['checkpoint_dir'], config['exp_name'],'codes')
    if not os.path.exists(save_code_path):
        os.makedirs(save_code_path)

    if if_save_code == 1:
        codes = {"q_codes": query_codes,
                'q_labels': query_labels,
                'g_codes' : gallery_codes,
                "g_labels" : gallery_labels}
        file_to_save = os.path.join(save_code_path, fname)
        np.save(file_to_save, codes)

    
    return mAP