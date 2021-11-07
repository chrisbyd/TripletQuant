import  os
import os.path as osp 

def results_to_txt(res,filename, model_name,sheet_name):
    save_dir = osp.join('./save','results')
    if not osp.exists(save_dir):
        os.makedirs(save_dir)
    filename = '{}_{}_results.txt'.format(filename,sheet_name)
    filename = osp.join(save_dir,filename)
    res = [str(item) + ' ' if index != len(res) - 1 else str(item) + '\n' for index, item in enumerate(res)]
    res = ''.join(res)
    res = model_name + ' '+ res
    with open(filename, 'a') as f:
        f.write(res)
