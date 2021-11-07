from .cross_transformer import CrossFormer
from .alexnet import AlexNet
import torch
import torch.nn as nn


EMBED_DIM = 96
DEPTHS = [ 2, 2, 6, 2 ]
NUM_HEADS = [ 3, 6, 12, 24 ]
GROUP_SIZE = [ 7, 7, 7, 7 ]
PATCH_SIZE = [4, 8, 16, 32]
MERGE_SIZE = [[2, 4], [2,4], [2, 4]]


CP_PATH = './checkpoint/crossformer-s.pth'

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.config = config
        self.transformer = Transformer()
        self.feature = nn.Linear(768, self.config['len_code'] * self.config['n_book'])
    
    def forward(self,input):
        features = self.transformer(input)
        features = self.feature(features)
        return features
    
    def initialize(self):
        self.transformer.load_checkpoint(self.config['pretrained_file'])
    



class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.encoder = CrossFormer(img_size = 224,
                                patch_size = PATCH_SIZE,
                                in_chans = 3,
                                embed_dim = EMBED_DIM,
                                depths = DEPTHS,
                                num_heads = NUM_HEADS,
                                group_size = GROUP_SIZE,
                                mlp_ratio= 4.0,
                                qkv_bias= True,
                                qk_scale= None,
                                drop_rate= 0.0,
                                drop_path_rate = 0.1,
                                ape= False,
                                patch_norm= True,
                                use_checkpoint = False,
                                merge_size = MERGE_SIZE)
    
    def forward(self, input):
        features = self.encoder.forward_features(input)
        return features

    def load_checkpoint(self, checkpoint_path):
        cpoint = torch.load(checkpoint_path)
      
        self.encoder.load_state_dict(cpoint['model'])
        print("Successfully loaded the encoder from checkpoint path {}".format(checkpoint_path))

