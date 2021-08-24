'''Add all arguments here'''

import argparse

'''_author = Yvan Tamdjo'''

arg = argparse.Namespace(
    # output dimension
    d_model = 512,

    # number of encoder or decoder stack
    N = 6,

    # number of head in attention
    h = 8,

    # queries dimension
    d_k = 64,

    # values dimension
    d_v = 64,

    # inner-layer dimension
    d_ff = 2048,

    # warmup step for optimization
    warmup_steps = 4000,

    # number of step during train
    num_step = 100000,

    # adam hyperparameter
    beta_1 = 0.9,
    beta_2 = 0.98,
    epsilon = 1e-9,

    # dropout
    dropout = 0.1,
)