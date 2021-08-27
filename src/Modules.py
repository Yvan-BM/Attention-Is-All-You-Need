import math
import torch
import numpy as np
import torch.nn as nn
import spacy

'''_author = Yvan Tamdjo'''

class ScaledDotProductAttention(nn.Module):

  def __init__(self):
    super(ScaledDotProductAttention, self).__init__()
    self.softmax = nn.Softmax()

  def forward(self, q, k, v, mask=None, e=1e-12):
      
    batch_size, head, length, d_model = k.size()

    attn = torch.matmul(q / math.sqrt(d_model), k.transpose(2, 3))


    if mask is not None:
        attn = attn.masked_fill(mask == 0, -e)

    attn = self.softmax(attn)

    v = torch.matmul(attn, v)

    return v, attn


class Tokenizer:

  def __init__(self):
    self.spacy_de = spacy.load('de')
    self.spacy_en = spacy.load('en')

  def tokenize_de(self, text):
    """
    Tokenizes German text from a string into a list of strings
    """
    return [tok.text for tok in self.spacy_de.tokenizer(text)]

  def tokenize_en(self, text):
    """
    Tokenizes English text from a string into a list of strings
    """
    return [tok.text for tok in self.spacy_en.tokenizer(text)]


class TokenEmbedding(nn.Embedding):
  """
  Token Embedding using torch.nn
  they will dense representation of word using weighted matrix
  """

  def __init__(self, vocab_size, d_model):
    super(TokenEmbedding, self).__init__(vocab_size, d_model, padding_idx=1)


class PostionalEncoding(nn.Module):
  """
  compute sinusoid encoding.
  """

  def __init__(self, d_model, max_len, device):
    super(PostionalEncoding, self).__init__()

    self.encoding = torch.zeros(max_len, d_model, device=device)
    self.encoding.requires_grad = False 

    pos = torch.arange(0, max_len, device=device).float().unsqueeze(dim=1)

    indice = torch.arange(0, d_model, step=2, device=device).float()

    self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (indice / d_model)))
    self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (indice / d_model)))

  def forward(self, x):
    batch_size, seq_len = x.size()

    return self.encoding[:seq_len, :]
