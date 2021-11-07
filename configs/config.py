from sacred import Experiment

ex =  Experiment("ap_quant")


@ex.config
def config():
    exp_name = 'ap_baseline'
    seed = 0
    dataset_name = 'cifar10'  # flickr coco nuswide and iaprtc
    batch_size = 30
    test_batch_size = 500
    
    resize_size = 256
    crop_size = 224

    machine_name = '2080'
 


    #optimizer config
    mode = 'train'


    momentum = 0.9
    weight_decay = 0. 
    epoch = 100

    dis_metric = 'pqdist'
    feat_dim = 128
    
    

     
    #evaluate interval
    eval_interval = 20
    
 

    #log config
   # pretrained_file = './pretrained_dir/Origin_ViT-B_32.pth'
    pretrained_file = './pretrained/crossformer-s.pth'
    log_dir = './save/logs'
    checkpoint_dir = './save/checkpoints'
    log_interval = 10
    val_check_interval =  313

    #from checkpoint
    topK = 54000

 

    #directories
    data_root = './data'
    n_class = 10
    len_code = 32
    n_book = 8
    alpha = 20.0
    beta = 4
    bn_word = 2
    lam_1 = 0.1
    lam_2 = 0.1 
    intn_word = pow(2, bn_word)




    #config for testing
    use_p_code = False










    test_only = False
