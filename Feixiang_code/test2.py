import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from sklearn.model_selection import train_test_split

# 设备设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 设置数据路径
path = "D:/dataset/COMP5703/dataset3"
os.chdir(path)
files_name = os.listdir(path)
model_files = []
data_files = []
files = []

for file_name in files_name:
    if file_name.endswith("model.csv"):
        model_files.append(file_name)
        files.append(file_name[:-9])
    else:
        data_files.append(file_name)

# 选取需要的特征
# feature_list = ["LHipAngles", "LFootProgressAngles", "RFootProgressAngles", "RHipAngles", "LKneeMoment", "RKneeMoment"]
feature_list = ["KneeMoment"]
features = [f"{element}_{axis}" for element in feature_list for axis in ["Y"]]


def read_csv(filepath, feature):
    return pd.read_csv(filepath).loc[:, feature].values


class MotionDataset(Dataset):
    def __init__(self, files_lst, attr):
        self.files = files_lst
        self.features = attr

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data_input = pd.read_csv(self.files[idx] + "trajectory.csv").iloc[:, 1:].values  # 读取输入
        data_output = read_csv(self.files[idx] + "model.csv", self.features)  # 读取目标输出
        tensor_data_input = torch.tensor(data_input, dtype=torch.float32)
        tensor_data_output = torch.tensor(data_output, dtype=torch.float32)
        return tensor_data_input, tensor_data_output


def collate_fn(batch):
    inputs, labels = zip(*batch)
    inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=0)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=0)
    input_lengths = torch.tensor([x.shape[0] for x in inputs])
    return inputs_padded, labels_padded, input_lengths


# 创建数据集
dataset = MotionDataset(files, features)

# 划分训练集和测试集
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# 创建数据加载器
batch_size = 4
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)


# 定义 BiLSTM 模型
class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)  # hidden_size * 2 Fit Bi-LSTM

    def forward(self, x, lengths):
        packed_input = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        output = self.fc(output)
        return output


# 初始化模型参数
input_size = pd.read_csv(files[0] + "trajectory.csv").shape[1] - 1  # 排除 Trajectory 列
hidden_size = 128
num_layers = 2
output_size = len(features)

# **模型迁移到 GPU**
model = BiLSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, output_size=output_size).to(device)

# **损失函数迁移到 GPU**
criterion = nn.MSELoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# **训练**
epochs = 25
for epoch in range(epochs):
    model.train()  # 设置为训练模式
    total_loss = 0.0
    for batch in train_dataloader:

        # **将数据迁移到 GPU**
        inputs, labels, lengths = batch
        inputs, labels, lengths = inputs.to(device), labels.to(device), lengths.to(device)

        optimizer.zero_grad()
        outputs = model(inputs, lengths)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / len(train_dataloader):.4f}")

# **测试**
model.eval()  # 设置为评估模式
total_test_loss = 0.0
with torch.no_grad():
    for batch in test_dataloader:
        inputs, labels, lengths = batch
        inputs, labels, lengths = inputs.to(device), labels.to(device), lengths.to(device)
        outputs = model(inputs, lengths)
        loss = criterion(outputs, labels)
        total_test_loss += loss.item()

# 输出测试集的损失
print(f"Test Loss: {total_test_loss / len(test_dataloader):.4f}")
