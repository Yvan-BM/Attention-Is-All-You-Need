import torch
import torch.nn as nn
import numpy as np
from config import arg
from Transformers.SubLayers import MultiHeadAttention, PositionwiseFeedForward

'''_author = Yvan Tamdjo'''

class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_hid, n_head, d_k, d_v, dropout=arg.dropout):
        super(EncoderLayer, self).__init__()
        self.MHA = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.layerNorm = nn.LayerNorm(d_model, eps=1e-6)
        self.FFN = PositionwiseFeedForward(d_model, d_hid, dropout=dropout)

    def forward(self, encoder_input, mask=None):
        res1 = encoder_input
        output1, MHA = self.MHA(encoder_input, encoder_input, encoder_input, mask=mask)
        # residual and normalization
        output1 = res1 + self.layerNorm(output1)
        res2 = output1
        output2 = self.FFN(output1)
        # residual and normalization
        output2 = res2 + self.layerNorm(output2)
        encoder_output = output2
        return encoder_output, MHA


class DecoderLayer(nn.Module):
    ''' Compose with three layers '''

    def __init__(self, d_model, d_hid, n_head, d_k, d_v, dropout=arg.dropout):
        super(DecoderLayer, self).__init__()
        self.MMHA = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.MHA = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.layerNorm = nn.LayerNorm(d_model, eps=1e-6)
        self.FFN = PositionwiseFeedForward(d_model, d_hid, dropout=dropout)

    def forward(self, decoder_input, encoder_output, self_attn_mask=None, dec_enc_attn_mask=None):
        res1 = decoder_input
        output1, MMHA = self.MMHA(decoder_input, decoder_input, decoder_input, mask=self_attn_mask)
        # residual and normalization
        output1 = res1 + self.layerNorm(output1)
        res2 = output1
        output2, MHA = self.MHA(output1, encoder_output, encoder_output, mask=dec_enc_attn_mask)
        # residual and normalization
        output2 = res2 + self.layerNorm(output2)
        res3 = output2
        output3 = self.FFN(output2)
        # residual and normalization
        decoder_output = res3 + output3

        return decoder_output, MMHA, MHA