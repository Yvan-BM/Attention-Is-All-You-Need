import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from Transformers.Modules import ScaledDotProductAttention

'''_author = Yvan Tamdjo'''

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_q = nn.Linear(d_model, n_head * d_k)
        self.w_k = nn.Linear(d_model, n_head * d_k)
        self.w_v = nn.Linear(d_model, n_head * d_v)
        self.FC = nn.Linear(n_head * d_v, d_model)

        self.attention = ScaledDotProductAttention(d_k)

        self.dropout = nn.Dropout(dropout)
        self.layerNorm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, Q, K, V, mask=None):

        residual = Q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        Q = self.w_q(Q).view(Q.size(0), Q.size(1), self.n_head, self.d_k)
        K = self.w_k(K).view(K.size(0), K.size(1), self.n_head, self.d_k)
        V = self.w_v(V).view(V.size(0), V.size(1), self.n_head, self.d_v)

        # Transpose for attention dot product: b x n x lq x dv
        Q, K, V = Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        Q, attn = self.attention(Q, K, V, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        Q = Q.transpose(1, 2).contiguous().view(Q.size(0), Q.size(1), -1)
        Q = self.FC(Q)
        Q = self.dropout(Q)

        return Q, attn


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_model, d_hid, dropout=0.1):
        super().__init__()
        self.W_1 = nn.Linear(d_model, d_hid)
        self.W_2 = nn.Linear(d_hid, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.W_1(x)
        x = F.relu(x)
        x = self.W_2(x)
        x = self.dropout(x)

        return x