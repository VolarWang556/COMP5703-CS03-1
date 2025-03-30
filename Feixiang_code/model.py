import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# 定义 Transformer 模型
class TransformerModel(nn.Module):
    def __init__(self, input_size, d_model, nhead, num_layers, output_size, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.d_model = d_model

        self.input_fc = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc_out = nn.Linear(d_model, output_size)

    def forward(self, src, lengths):
        # src: [batch_size, seq_len, input_size]
        src = self.input_fc(src)  # [batch_size, seq_len, d_model]

        src = self.pos_encoder(src)

        batch_size, seq_len, _ = src.size()
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=src.device)
        for i, length in enumerate(lengths):
            mask[i, :length] = False

        output = self.transformer_encoder(src, src_key_padding_mask=mask)
        output = self.fc_out(output)
        return output
