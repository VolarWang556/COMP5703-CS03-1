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
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_rate=0.5):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)  # 输出维度乘以2符合双向LSTM
        self.dropout = nn.Dropout(dropout_rate)
        self.layernorm = nn.LayerNorm(output_size)  # LayerNorm作用于最后一维

    def forward(self, x, lengths):
        # 对变长序列进行 pack 和 pad 操作
        packed_input = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        # 通过线性层
        output = self.fc(output)
        # 应用 Dropout
        output = self.dropout(output)
        # 直接对最后一维进行归一化（LayerNorm 默认作用于最后一维）
        output = self.layernorm(output)
        return output



# 初始化模型参数
input_size = pd.read_csv(files[0] + "trajectory.csv").shape[1] - 1  # 排除 Trajectory 列
hidden_size = 64
num_layers = 2
output_size = len(features)

# **模型迁移到 GPU**
model = BiLSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, output_size=output_size).to(device)

# **损失函数迁移到 GPU**
# criterion = nn.MSELoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

alpha = 0.05  # 可根据需要调整


class SegmentedMaxLoss(nn.Module):
    def __init__(self, alpha=0.5, reduction='mean'):
        super().__init__()
        self.alpha = alpha  # 分段损失的权重系数
        self.mse_loss = nn.MSELoss(reduction=reduction)  # 基础MSE损失函数[1](@ref)
        self.reduction = reduction

    def forward(self, outputs, labels, lengths):
        base_loss = self.mse_loss(outputs, labels)

        batch_size = outputs.size(0)
        max_loss_first, max_loss_second = 0.0, 0.0

        # 遍历每个样本处理变长序列
        for i in range(batch_size):
            seq_len = lengths[i].item()
            if seq_len < 2:
                continue  # 跳过长度不足的序列

            half = seq_len // 2
            out_seq = outputs[i, :seq_len, :]  # (seq_len, features)
            lab_seq = labels[i, :seq_len, :]

            # 前半段最大值误差（向量化计算）
            max_out_first, _ = torch.max(out_seq[:half], dim=0)  # (features,)
            max_lab_first, _ = torch.max(lab_seq[:half], dim=0)
            loss_first = torch.mean((max_out_first - max_lab_first) ** 2)

            # 后半段最大值误差
            max_out_second, _ = torch.max(out_seq[half:], dim=0)
            max_lab_second, _ = torch.max(lab_seq[half:], dim=0)
            loss_second = torch.mean((max_out_second - max_lab_second) ** 2)

            max_loss_first += loss_first
            max_loss_second += loss_second

        # 平均分段损失
        if batch_size > 0:
            max_loss_first /= batch_size
            max_loss_second /= batch_size

        # 总损失 = 基础损失 + 加权分段损失[7](@ref)
        total_loss = base_loss + self.alpha * (max_loss_first + max_loss_second)
        return total_loss
epochs = 2
criterion = SegmentedMaxLoss(alpha).to(device)
# 在训练循环中替换原来的损失计算，如下：
for epoch in range(epochs):
    model.train()  # 训练模式
    total_loss = 0.0
    for batch in train_dataloader:
        inputs, labels, lengths = batch
        inputs, labels, lengths = inputs.to(device), labels.to(device), lengths.to(device)

        optimizer.zero_grad()
        outputs = model(inputs, lengths)
        # 使用自定义损失函数计算总损失
        loss = criterion(outputs, labels, lengths)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / len(train_dataloader):.4f}")

import matplotlib.pyplot as plt

# 可视化一个样本的真实值与预测值
model.eval()
with torch.no_grad():
    # 拿一个 batch 中第一个样本
    for batch in test_dataloader:
        inputs, labels, lengths = batch
        inputs, labels, lengths = inputs.to(device), labels.to(device), lengths.to(device)
        outputs = model(inputs, lengths)

        # 只看第一个样本
        pred = outputs[0].cpu().numpy()
        true = labels[0].cpu().numpy()

        # 截取到有效长度，避免 padding 的影响
        valid_len = lengths[0].item()
        pred = pred[:valid_len, 0]  # 取第一维度值
        true = true[:valid_len, 0]

        # 绘图
        plt.figure(figsize=(10, 5))
        plt.plot(true, label="True", color="blue")
        plt.plot(pred, label="Predicted", color="red", linestyle='--')
        plt.title("True vs Predicted Output")
        plt.xlabel("Time Step")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        break  # 只画一个 batch 就够了

