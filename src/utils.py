import torch.nn as nn

'''_author = Yvan Tamdjo'''

def count_parameters(model):
  return sum(p.numel() for p in model.parameters() if p.requires_grad)


def initialize_weights(m):
  if hasattr(m, 'weight') and m.weight.dim() > 1:
    nn.init.kaiming_uniform(m.weight.data)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs