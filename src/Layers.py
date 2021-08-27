import torch
import torch.nn as nn
import numpy as np
from config import arg
from Transformers.SubLayers import MultiHeadAttention, PositionwiseFeedForward

'''_author = Yvan Tamdjo'''

class EncoderLayer(nn.Module):

  def __init__(self, d_model, d_hid, n_head, dropout):
    super(EncoderLayer, self).__init__()
    self.attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
    self.layerNorm1 = nn.LayerNorm(d_model, eps=1e-6)
    self.dropout1 = nn.Dropout(dropout)

    self.ffn = PositionwiseFeedForward(d_model=d_model, d_hid=d_hid, dropout=dropout)
    self.layerNorm2 = nn.LayerNorm(d_model, eps=1e-6)
    self.dropout2 = nn.Dropout(dropout)

  def forward(self, x, s_mask):
    res1 = x
    x = self.attention(Q=x, K=x, V=x, mask=s_mask)
    x = self.layerNorm1(x + res1)
    x = self.dropout1(x)

    res2 = x
    x = self.ffn(x)
    x = self.layerNorm2(x + res2)
    x = self.dropout2(x)
    return x


class DecoderLayer(nn.Module):

  def __init__(self, d_model, d_hid, n_head, dropout):
    super(DecoderLayer, self).__init__()
    self.self_attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
    self.layerNorm1 = nn.LayerNorm(d_model, eps=1e-6)
    self.dropout1 = nn.Dropout(dropout)

    self.enc_dec_attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
    self.layerNorm2 = nn.LayerNorm(d_model, eps=1e-6)
    self.dropout2 = nn.Dropout(dropout)

    self.ffn = PositionwiseFeedForward(d_model=d_model, d_hid=d_hid, dropout=dropout)
    self.layerNorm3 = nn.LayerNorm(d_model, eps=1e-6)
    self.dropout3 = nn.Dropout(dropout)

  def forward(self, dec, enc, t_mask, s_mask):
    res1 = dec
    x = self.self_attention(Q=dec, K=dec, V=dec, mask=t_mask)
    x = self.layerNorm1(x + res1)
    x = self.dropout1(x)

    if enc is not None:
      res2 = x
      x = self.enc_dec_attention(Q=x, K=enc, V=enc, mask=s_mask)
      x = self.layerNorm2(x + res2)
      x = self.dropout2(x)

    res3 = x
    x = self.ffn(x)
    x = self.layerNorm3(x + res3)
    x = self.dropout3(x)
    return x