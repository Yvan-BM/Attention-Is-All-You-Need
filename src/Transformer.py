import torch
import torch.nn as nn
import numpy as np
from src.Layers import EncoderLayer, DecoderLayer
from src.Modules import TokenEmbedding, PostionalEncoding

'''_author = Yvan Tamdjo'''

class Encoder(nn.Module):

  def __init__(self, enc_voc_size, max_len, d_model, d_hid, n_head, n_layers, dropout, device):
    super().__init__()
    
    self.token_emb = TokenEmbedding(enc_voc_size, d_model)
    self.pos_emb = PostionalEncoding(d_model, max_len, device)
    self.dropout = nn.Dropout(dropout)

    self.layers = nn.ModuleList([EncoderLayer(d_model=d_model,
                                              d_hid=d_hid,
                                              n_head=n_head,
                                              dropout=dropout)
                                  for _ in range(n_layers)])

  def forward(self, x, s_mask):
    x = self.dropout(self.token_emb(x) + self.pos_emb(x))

    for layer in self.layers:
      x = layer(x, s_mask)

    return x


class Decoder(nn.Module):
  def __init__(self, dec_voc_size, max_len, d_model, d_hid, n_head, n_layers, dropout, device):
    super().__init__()
    
    self.token_emb = TokenEmbedding(dec_voc_size, d_model)
    self.pos_emb = PostionalEncoding(d_model, max_len, device)
    self.dropout = nn.Dropout(dropout)

    self.layers = nn.ModuleList([DecoderLayer(d_model=d_model,
                                              d_hid=d_hid,
                                              n_head=n_head,
                                              dropout=dropout)
                                  for _ in range(n_layers)])

    self.linear = nn.Linear(d_model, dec_voc_size)

  def forward(self, trg, enc_src, trg_mask, src_mask):
    trg = self.dropout(self.token_emb(trg) + self.pos_emb(trg))

    for layer in self.layers:
      trg = layer(trg, enc_src, trg_mask, src_mask)

    output = self.linear(trg)
    return output


class Transformer(nn.Module):

  def __init__(self, src_pad_idx, trg_pad_idx, trg_sos_idx, enc_voc_size, dec_voc_size, d_model, n_head, max_len,
                d_hid, n_layers, dropout, device):
    super().__init__()
    self.src_pad_idx = src_pad_idx
    self.trg_pad_idx = trg_pad_idx
    self.trg_sos_idx = trg_sos_idx
    self.device = device
    self.encoder = Encoder(d_model=d_model,
                            n_head=n_head,
                            max_len=max_len,
                            d_hid=d_hid,
                            enc_voc_size=enc_voc_size,
                            dropout=dropout,
                            n_layers=n_layers,
                            device=device)

    self.decoder = Decoder(d_model=d_model,
                            n_head=n_head,
                            max_len=max_len,
                            d_hid=d_hid,
                            dec_voc_size=dec_voc_size,
                            dropout=dropout,
                            n_layers=n_layers,
                            device=device)

  def forward(self, src, trg):
    src_mask = self.make_pad_mask(src, src)

    src_trg_mask = self.make_pad_mask(trg, src)

    trg_mask = self.make_pad_mask(trg, trg) * \
                self.make_no_peak_mask(trg, trg)

    enc_src = self.encoder(src, src_mask)
    output = self.decoder(trg, enc_src, trg_mask, src_trg_mask)
    return output

  def make_pad_mask(self, q, k):
    len_q, len_k = q.size(1), k.size(1)

    # batch_size x 1 x 1 x len_k
    k = k.ne(self.src_pad_idx).unsqueeze(1).unsqueeze(2)
    # batch_size x 1 x len_q x len_k
    k = k.repeat(1, 1, len_q, 1)

    # batch_size x 1 x len_q x 1
    q = q.ne(self.src_pad_idx).unsqueeze(1).unsqueeze(3)
    # batch_size x 1 x len_q x len_k
    q = q.repeat(1, 1, 1, len_k)

    mask = k & q
    return mask

  def make_no_peak_mask(self, q, k):
    len_q, len_k = q.size(1), k.size(1)

    # len_q x len_k
    mask = torch.tril(torch.ones(len_q, len_k)).type(torch.BoolTensor).to(self.device)

    return mask