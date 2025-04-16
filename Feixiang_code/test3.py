import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import matplotlib.pyplot as plt


# 设备设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# 数据读取及Dataset定义（这里假设每个样本是一个时间序列，
# 每个时间步对应一个点云数据，形状为 [C, N]，例如 [9, 2048]）
class MotionDataset(Dataset):
    def __init__(self, files_lst, attr):
        self.files = files_lst
        self.features = attr

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # 假设每个序列存于一个 CSV 中，行数代表时间步，
        # 每一行存放一个flatten后的点云数据，
        # 并能reshape成 [C, N]，这里以 [9, 2048] 为例
        data_input = pd.read_csv(self.files[idx] + "trajectory.csv").iloc[:, 1:].values
        # 假设目标输出依然来自model.csv
        data_output = pd.read_csv(self.files[idx] + "model.csv").loc[:, self.features].values
        # 假设data_input每行可以reshape成 [9, 2048]：
        # 得到时间步数 T = data_input.shape[0]
        T = data_input.shape[0]
        # 这里假设通道数为9，点数为2048
        tensor_sequence = []
        for t in range(T):
            # 调整每一行到形状 [9, 2048]
            point_cloud = torch.tensor(data_input[t], dtype=torch.float32)
            point_cloud = point_cloud.view(9, -1)  # 自动计算点数
            tensor_sequence.append(point_cloud)
        tensor_sequence = torch.stack(tensor_sequence)  # shape: [T, 9, N]
        tensor_output = torch.tensor(data_output, dtype=torch.float32)  # 可根据任务自行处理
        return tensor_sequence, tensor_output


def collate_fn(batch):
    # 每个样本是 (sequence [T, 9, N], label)
    sequences, labels = zip(*batch)
    # 使用 pad_sequence 对序列进行padding，要求序列为 list of [T, 9, N] 张量
    sequences_padded = pad_sequence(sequences, batch_first=True, padding_value=0)  # [B, T_max, 9, N]
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=0)
    lengths = torch.tensor([seq.shape[0] for seq in sequences])
    return sequences_padded, labels_padded, lengths


# 设置数据路径和文件列表（示例）
path = "D:/dataset/COMP5703/dataset3"
os.chdir(path)
files_name = os.listdir(path)
files = []
for file_name in files_name:
    if file_name.endswith("model.csv"):
        # 假设文件命名规则相同
        files.append(file_name[:-9])

feature_list = ["KneeMoment"]  # 示例目标特征
features = [f"{element}_{axis}" for element in feature_list for axis in ["Y"]]

# 创建数据集与 DataLoader
dataset = MotionDataset(files, features)
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

batch_size = 4
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)


############################
# 定义预训练 PointNet 模块 #
############################
# 示例：使用 PointNetEncoder（输出全局特征向量）
class STN3d(nn.Module):
    # 此处为占位，需用实际定义
    def __init__(self, channel):
        super(STN3d, self).__init__()
        # 定义STN结构...
        self.fc = nn.Linear(channel, channel)  # 简单示例

    def forward(self, x):
        B, C, N = x.shape
        # 返回单位矩阵作为示例
        return torch.eye(C).unsqueeze(0).repeat(B, 1, 1).to(x.device)


class PointNetEncoder(nn.Module):
    def __init__(self, global_feat=True, feature_transform=False, channel=9):
        super(PointNetEncoder, self).__init__()
        self.stn = STN3d(channel)
        self.conv1 = nn.Conv1d(channel, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform

    def forward(self, x):
        # x: [B, 9, N]
        B, D, N = x.size()
        trans = self.stn(x)  # [B, 9, 9]
        x = x.transpose(2, 1)  # [B, N, 9]
        x = torch.bmm(x, trans)  # 对齐
        x = x.transpose(2, 1)  # [B, 9, N]
        x = F.relu(self.bn1(self.conv1(x)))  # [B, 64, N]
        x = F.relu(self.bn2(self.conv2(x)))  # [B, 128, N]
        x = self.bn3(self.conv3(x))  # [B, 1024, N]
        x = torch.max(x, 2, keepdim=True)[0]  # [B, 1024, 1]
        x = x.view(B, 1024)  # [B, 1024]
        return x, trans, None


############################
# 定义预训练 BiLSTM 模块 #
############################
class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_size * 2, output_size)  # 双向输出

    def forward(self, x, lengths):
        # x: [B, T, input_size]
        packed_input = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        # output = self.dropout(output)
        output = self.fc(output)
        return output


##############################################
# 定义组合模型：将 PointNet 和 BiLSTM 拼接 #
##############################################
class FullModel(nn.Module):
    def __init__(self, pointnet, lstm_model):
        super(FullModel, self).__init__()
        self.pointnet = pointnet  # 预训练 PointNet
        self.lstm = lstm_model  # 预训练 BiLSTM

    def forward(self, x, lengths):
        # x: [B, T, C, N]
        B, T, C, N = x.shape
        # 合并 batch 与时间维度，便于并行处理
        x_reshaped = x.view(B * T, C, N)  # [B*T, C, N]
        # 获取每个时间步的全局特征，假设输出维度为1024
        pointnet_feat, _, _ = self.pointnet(x_reshaped)  # [B*T, 1024]
        # 恢复为序列： [B, T, 1024]
        feat_seq = pointnet_feat.view(B, T, -1)
        # 输入到 BiLSTM
        output = self.lstm(feat_seq, lengths)
        return output


##############################################
# 初始化各模块，并加载预训练权重 #
##############################################
# 假设预训练权重文件存在
pointnet = PointNetEncoder(global_feat=True, feature_transform=False, channel=9).to(device)
lstm_input_size = 1024  # 与PointNet全局特征输出匹配
hidden_size = 64
num_layers = 2
output_size = len(features)  # 根据目标输出设定

bilstm = BiLSTM(input_size=lstm_input_size, hidden_size=hidden_size, num_layers=num_layers, output_size=output_size).to(
    device)

# 加载预训练权重（确保权重文件格式与保存时一致）：
# 加载 PointNet 权重
if os.path.exists("pointnet.pth"):
    checkpoint_pointnet = torch.load("pointnet.pth", map_location=device)
    # 如果保存的是checkpoint字典，则加载其中预训练的 state_dict 部分
    pointnet.load_state_dict(checkpoint_pointnet["model_state_dict"])
    print("Loaded PointNet pre-trained weights.")

# 加载 BiLSTM 权重
if os.path.exists("bilstm.pth"):
    checkpoint_lstm = torch.load("bilstm.pth", map_location=device)
    bilstm.load_state_dict(checkpoint_lstm["model_state_dict"])
    print("Loaded BiLSTM pre-trained weights.")

# 组合为 FullModel
model = FullModel(pointnet, bilstm).to(device)

##############################################
# 微调训练设置（示例） #
##############################################
# 微调训练设置（示例）
criterion = nn.MSELoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 50

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

