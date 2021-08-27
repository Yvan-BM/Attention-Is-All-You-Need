'''Add all arguments here'''
import torch

import argparse

'''_author = Yvan Tamdjo'''

arg = argparse.Namespace(

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),

    batch_size = 64,

    max_len = 256,

    # output dimension
    d_model = 512,

    # number of encoder or decoder stack
    n_layers = 6,

    # number of head in attention
    n_heads = 8,

    # inner-layer dimension
    d_hid = 2048,

    # warmup step for optimization
    warmup = 100,

    # number of step during train
    epoch = 30,

    # adam hyperparameter
    adam_eps = 5e-9,

    # dropout
    dropout = 0.1,

    init_lr = 1e-5,

    factor = 0.9,

    
    patience = 10,

    clip = 1.0,

    weight_decay = 5e-4,

    inf = float('inf')
)