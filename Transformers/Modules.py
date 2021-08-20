import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, d_k, dropout=0.1):
        super().__init__()
        self.d_k = d_k
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / math.sqrt(self.d_k), k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn


