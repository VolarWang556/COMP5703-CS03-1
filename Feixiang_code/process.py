import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import tqdm

path = "D:/dataset/COMP5703/dataset"
os.chdir(path)
files_name = os.listdir(path)
model_files = []
data_files = []
files = []

for file_name in files_name:
    if file_name[-9:] == "model.csv":
        model_files.append(file_name)
        files.append(file_name[:-9])
    else:
        data_files.append(file_name)
        
s = "sdfsdfsdfsd"

# Find all related attr in model.csv file


feature_list = ["LHipAngles", "LFootProgressAngles", "RFootProgressAngles", "RHipAngles", "LKneeMoment", "RKneeMoment"]
features = []
for element in feature_list:
    features.append(element + "_X")
    features.append(element + "_Y")
    features.append(element + "_Z")
    # attr.append(element + "_X'") # First derivative
    # attr.append(element + "_Y'")
    # attr.append(element + "_Z'")
    # attr.append(element + "_X''") # Second derivative
    # attr.append(element + "_Y''")
    # attr.append(element + "_Z''")
def read_csv(filepath, feature):
    arr = pd.read_csv(filepath).loc[:, feature].values
    return arr


class MotionDataset(Dataset):
    def __init__(self, files_lst, attr):
        self.files = files_lst
        self.features = attr

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data1 = read_csv(self.files[idx] + "model.csv", self.features)  # 读取词向量
        data2 = pd.read_csv(self.files[idx] + "trajectory.csv").iloc[:, 1:].values
        tensor_data_output = torch.tensor(data1, dtype=torch.float32)  # 转换为张量
        tensor_data_input = torch.tensor(data2, dtype=torch.float32)  # 转换为张量
        return tensor_data_input, tensor_data_output  # 返回 (句子长度, 词向量维度) 的张量


def collate_fn(batch):
    inputs, labels = zip(*batch)  # 解包 batch，把 inputs 和 labels 分开
    inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=0)  # 仅对 inputs 进行 padding
    return inputs_padded, labels


dataset = MotionDataset(files, features)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)


