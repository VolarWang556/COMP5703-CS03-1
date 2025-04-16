import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

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
# dataset = MotionDataset(files, features)

# # 划分训练集和测试集
# train_size = int(0.8 * len(dataset))
# test_size = len(dataset) - train_size
# train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
#
# # 创建数据加载器
# batch_size = 4
# train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
# test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

dataset = MotionDataset(files, features)
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

batch_size = 4
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)


# 定义 BiLSTM 模型
class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)  # hidden_size * 2 Fit Bi-LSTM
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, lengths):
        packed_input = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        output = self.dropout(output)
        output = self.fc(output)
        return output


# 初始化模型参数
input_size = pd.read_csv(files[0] + "trajectory.csv").shape[1] - 1  # 排除 Trajectory 列
hidden_size = 64
num_layers = 2
output_size = len(features)

# **模型迁移到 GPU**
model = BiLSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, output_size=output_size).to(device)

# **损失函数迁移到 GPU**
criterion = nn.MSELoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# **训练**
epochs = 50
# for epoch in range(epochs):
#     model.train()  # 设置为训练模式
#     total_loss = 0.0
#     for batch in train_dataloader:
#
#         # **将数据迁移到 GPU**
#
#         inputs, labels, lengths = batch
#         inputs, labels, lengths = inputs.to(device), labels.to(device), lengths.to(device)
#
#         optimizer.zero_grad()
#         outputs = model(inputs, lengths)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#
#         total_loss += loss.item()
#
#     print(f"Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / len(train_dataloader):.4f}")
#
# # **测试**
# model.eval()  # 设置为评估模式
# total_test_loss = 0.0
# with torch.no_grad():
#     for batch in test_dataloader:
#         inputs, labels, lengths = batch
#         inputs, labels, lengths = inputs.to(device), labels.to(device), lengths.to(device)
#         outputs = model(inputs, lengths)
#         loss = criterion(outputs, labels)
#         total_test_loss += loss.item()
#
# # 输出测试集的损失
# print(f"Test Loss: {total_test_loss / len(test_dataloader):.4f}")
#
# import matplotlib.pyplot as plt
#
# # 可视化一个样本的真实值与预测值
# model.eval()
# with torch.no_grad():
#     # 拿一个 batch 中第一个样本
#     for batch in test_dataloader:
#         inputs, labels, lengths = batch
#         inputs, labels, lengths = inputs.to(device), labels.to(device), lengths.to(device)
#         outputs = model(inputs, lengths)
#
#         # 只看第一个样本
#         pred = outputs[0].cpu().numpy()
#         true = labels[0].cpu().numpy()
#
#         # 截取到有效长度，避免 padding 的影响
#         valid_len = lengths[0].item()
#         pred = pred[:valid_len, 0]  # 取第一维度值
#         true = true[:valid_len, 0]
#
#         # 绘图
#         plt.figure(figsize=(10, 5))
#         plt.plot(true, label="True", color="blue")
#         plt.plot(pred, label="Predicted", color="red", linestyle='--')
#         plt.title("True vs Predicted Output")
#         plt.xlabel("Time Step")
#         plt.ylabel("Value")
#         plt.legend()
#         plt.grid(True)
#         plt.tight_layout()
#         plt.show()
#
#         break  # 只画一个 batch 就够了
#
for epoch in range(epochs):
    model.train()
    total_train_loss = 0.0
    for batch in train_dataloader:
        inputs, labels, lengths = batch
        inputs, labels, lengths = inputs.to(device), labels.to(device), lengths.to(device)

        optimizer.zero_grad()
        outputs = model(inputs, lengths)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_dataloader)

    # 验证阶段
    model.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        for batch in val_dataloader:
            inputs, labels, lengths = batch
            inputs, labels, lengths = inputs.to(device), labels.to(device), lengths.to(device)
            outputs = model(inputs, lengths)
            loss = criterion(outputs, labels)
            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(val_dataloader)
    print(f"Epoch [{epoch + 1}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

# 测试阶段
model.eval()
total_test_loss = 0.0
with torch.no_grad():
    for batch in test_dataloader:
        inputs, labels, lengths = batch
        inputs, labels, lengths = inputs.to(device), labels.to(device), lengths.to(device)
        outputs = model(inputs, lengths)
        loss = criterion(outputs, labels)
        total_test_loss += loss.item()
print(f"\nFinal Test Loss: {total_test_loss / len(test_dataloader):.4f}")

# 评估模型 + 特殊评估指标（误差均值）
model.eval()
total_test_loss = 0.0

front_half_true = []
front_half_pred = []
back_half_true = []
back_half_pred = []

with torch.no_grad():
    for batch in test_dataloader:
        inputs, labels, lengths = batch
        inputs, labels, lengths = inputs.to(device), labels.to(device), lengths.to(device)
        outputs = model(inputs, lengths)
        loss = criterion(outputs, labels)
        total_test_loss += loss.item()

        # 遍历 batch 中的每一个样本，统计最大值误差
        for i in range(inputs.size(0)):
            valid_len = lengths[i].item()
            pred = outputs[i, :valid_len, 0].cpu().numpy()
            true = labels[i, :valid_len, 0].cpu().numpy()

            half = valid_len // 2

            # 前50%
            front_true = true[:half]
            front_pred = pred[:half]
            if len(front_true) > 0:
                max_idx_front = np.argmax(front_true)
                front_half_pred.append(front_pred[max_idx_front])
                front_half_true.append(front_true[max_idx_front])

            # 后50%
            back_true = true[half:]
            back_pred = pred[half:]
            if len(back_true) > 0:
                max_idx_back = np.argmax(back_true)
                back_half_pred.append(back_pred[max_idx_back])
                back_half_true.append(back_true[max_idx_back])

# 输出结果
print(f"Test Loss (Overall MSE): {total_test_loss / len(test_dataloader):.4f}")
print(f"Front Half Max Pred Avg: {np.mean(front_half_pred):.4f}")
print(f"Front Half Max True Avg: {np.mean(front_half_true):.4f}")
print(f"Back Half Max Pred Avg: {np.mean(back_half_pred):.4f}")
print(f"Back Half Max True Avg: {np.mean(back_half_true):.4f}")


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

