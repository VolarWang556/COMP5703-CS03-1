import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

path = "D:/dataset/COMP5703/dataset"
os.chdir(path)
files_name = os.listdir(path)
model_files = []
data_files = []

for file_name in files_name:
    if file_name[-9:] == "model.csv":
        model_files.append(file_name)
    else:
        data_files.append(file_name)


class SentenceDataset(Dataset):
    def __init__(self, csv_dir):
        self.files = [os.path.join(csv_dir, f) for f in os.listdir(csv_dir) if f.endswith('.csv')]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = pd.read_csv(self.files[idx], header=None, skiprows=1).iloc[:, 1:].values  # 读取词向量
        tensor_data = torch.tensor(data, dtype=torch.float32)  # 转换为张量
        return tensor_data  # 返回 (句子长度, 词向量维度) 的张量


def collate_fn(batch):
    return pad_sequence(batch, batch_first=True, padding_value=0)  # 对齐不同长度的句子


dataset = SentenceDataset(path)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
