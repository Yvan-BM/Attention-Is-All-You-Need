import numpy as np
import torch.nn as nn
from config import arg
from src.Modules import ScaledDotProductAttention

'''_author = Yvan Tamdjo'''

class MultiHeadAttention(nn.Module):

  def __init__(self, d_model, n_head):
    super(MultiHeadAttention, self).__init__()

    self.n_head = n_head
    self.w_q = nn.Linear(d_model, d_model)
    self.w_k = nn.Linear(d_model, d_model)
    self.w_v = nn.Linear(d_model, d_model)
    self.w_concat = nn.Linear(d_model, d_model)
    self.attention = ScaledDotProductAttention()
    
  def forward(self, Q, K, V, mask=None):

    Q, K, V = self.w_q(Q), self.w_k(K), self.w_v(V)

    Q, K, V = self.split(Q), self.split(K), self.split(V)

    out, attention = self.attention(Q, K, V, mask=mask)

    out = self.concat(out)
    out = self.w_concat(out)

    return out

  def split(self, tensor):
  
    batch_size, length, d_model = tensor.size()

    d_tensor = d_model // self.n_head
    tensor = tensor.view(batch_size, self.n_head, length, d_tensor)
    # it is similar with group convolution (split by number of heads)

    return tensor

  def concat(self, tensor):
  
    batch_size, head, length, d_tensor = tensor.size()
    d_model = head * d_tensor

    tensor = tensor.view(batch_size, length, d_model)
    return tensor


class PositionwiseFeedForward(nn.Module):

  def __init__(self, d_model, d_hid, dropout=0.1):
    super(PositionwiseFeedForward, self).__init__()
    self.W_1 = nn.Linear(d_model, d_hid)
    self.W_2 = nn.Linear(d_hid, d_model)
    self.relu = nn.ReLU()
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    x = self.W_1(x)
    x = self.relu(x)
    x = self.dropout(x)
    x = self.W_2(x)
    return x