import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 设备设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 设置数据路径
path = "D:/dataset/COMP5703/dataset4"
os.chdir(path)
files_name = os.listdir(path)
model_files = []
data_files = []
files = []

for file_name in files_name:
    if file_name.endswith("model.csv"):
        model_files.append(file_name)
        files.append(file_name[:-9])  # 保留基础文件名
    else:
        data_files.append(file_name)

# 选取需要的特征
# feature_list = ["LHipAngles", "LFootProgressAngles", "RFootProgressAngles", "RHipAngles", "LKneeMoment", "RKneeMoment"]
feature_list = ["KneeMoment"]
features = [f"{element}_{axis}" for element in feature_list for axis in ["Y"]]


def read_csv(filepath, feature):
    return pd.read_csv(filepath).loc[:, feature].values


# 修改 MotionDataset，使其同时返回文件名
class MotionDataset(Dataset):
    def __init__(self, files_lst, attr):
        self.files = files_lst
        self.features = attr

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # 读取输入数据（trajectory.csv 从第2列开始）
        file_base = self.files[idx]
        data_input = pd.read_csv(file_base + "trajectory.csv").iloc[:, 1:].values
        # 读取目标输出（model.csv 中选取的特征）
        data_output = read_csv(file_base + "model.csv", self.features)
        tensor_data_input = torch.tensor(data_input, dtype=torch.float32)
        tensor_data_output = torch.tensor(data_output, dtype=torch.float32)
        # 同时返回文件名（可用于后续输出）
        return tensor_data_input, tensor_data_output, file_base


# 修改 collate_fn 同时收集文件名
def collate_fn(batch):
    inputs, labels, file_names = zip(*batch)
    inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=0)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=0)
    input_lengths = torch.tensor([x.shape[0] for x in inputs])
    return inputs_padded, labels_padded, input_lengths, file_names


# 创建数据集
dataset = MotionDataset(files, features)
# 划分数据集（训练集、验证集、测试集）
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# 为了最终输出所有样本的评估结果，将三个子集合并
combined_dataset = ConcatDataset([train_dataset, val_dataset, test_dataset])
combined_dataloader = DataLoader(combined_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)


# 定义 BiLSTM 模型
class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)  # Fit Bi-LSTM

    def forward(self, x, lengths):
        packed_input = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        output = self.fc(output)
        return output


# 初始化模型参数
input_size = pd.read_csv(files[0] + "trajectory.csv").shape[1] - 1  # 排除 Trajectory 列
hidden_size = 64
num_layers = 2
output_size = len(features)

# 模型迁移到 GPU
model = BiLSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, output_size=output_size).to(
    device)
criterion = nn.MSELoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
epochs = 50
for epoch in range(epochs):
    model.train()
    total_train_loss = 0.0
    for batch in DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn):
        inputs, labels, lengths, _ = batch
        inputs, labels, lengths = inputs.to(device), labels.to(device), lengths.to(device)
        optimizer.zero_grad()
        outputs = model(inputs, lengths)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_dataset)

    # 验证阶段
    model.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        for batch in DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn):
            inputs, labels, lengths, _ = batch
            inputs, labels, lengths = inputs.to(device), labels.to(device), lengths.to(device)
            outputs = model(inputs, lengths)
            loss = criterion(outputs, labels)
            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(val_dataset)
    print(f"Epoch [{epoch + 1}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

# 测试并输出全部样本的评估结果到 CSV 文件
model.eval()
results = []  # 用于存储每个样本的评估结果

with torch.no_grad():
    for batch in combined_dataloader:
        inputs, labels, lengths, file_names = batch
        inputs, labels, lengths = inputs.to(device), labels.to(device), lengths.to(device)
        outputs = model(inputs, lengths)

        # 遍历当前 batch 中的每个样本
        batch_size = inputs.size(0)
        for i in range(batch_size):
            valid_len = lengths[i].item()
            # 注意这里取输出及真实值的第一维度（假设只有一个特征，即 features 长度为 1）
            pred_seq = outputs[i, :valid_len, 0].cpu().numpy()
            true_seq = labels[i, :valid_len, 0].cpu().numpy()
            half = valid_len // 2

            # 计算前50%（若数据为空则使用 nan）
            if half > 0:
                front_true = true_seq[:half]
                front_pred = pred_seq[:half]
                front_max_true = float(front_true[np.argmax(front_true)])
                front_max_pred = float(front_pred[np.argmax(front_true)])
            else:
                front_max_true = np.nan
                front_max_pred = np.nan

            # 计算后50%
            if valid_len - half > 0:
                back_true = true_seq[half:]
                back_pred = pred_seq[half:]
                back_max_true = float(back_true[np.argmax(back_true)])
                back_max_pred = float(back_pred[np.argmax(back_true)])
            else:
                back_max_true = np.nan
                back_max_pred = np.nan

            results.append({
                "FileName": file_names[i],
                "Front_Half_Max_True": front_max_true,
                "Front_Half_Max_Pred": front_max_pred,
                "Back_Half_Max_True": back_max_true,
                "Back_Half_Max_Pred": back_max_pred
            })
# 可视化测试集中的一个样本的真实值与预测值曲线图
model.eval()
with torch.no_grad():
    # 从 test_dataloader 中取出一个 batch
    for batch in DataLoader(test_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn):
        inputs, labels, lengths = batch
        # 将数据迁移到设备（如果需要 GPU 的话）
        inputs, labels, lengths = inputs.to(device), labels.to(device), lengths.to(device)
        outputs = model(inputs, lengths)

        # 取 batch 中第一个样本
        pred = outputs[0].cpu().numpy()
        true = labels[0].cpu().numpy()

        # 取出有效长度，避免 padding 部分干扰
        valid_len = lengths[0].item()
        pred = pred[:valid_len, 0]  # 假设只关注特征的第一维（如只有一个特征时）
        true = true[:valid_len, 0]

        # 绘制曲线图
        plt.figure(figsize=(10, 5))
        plt.plot(true, label="True", color="blue")
        plt.plot(pred, label="Predicted", color="red", linestyle="--")
        plt.title("Test Sample: True vs Predicted")
        plt.xlabel("Time Step")
        plt.ylabel("Value")

        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # 只绘制一个 batch 中的一个样本
        break

# 保存 CSV 文件
results_df = pd.DataFrame(results)
path = "D:/dataset/COMP5703"
os.chdir(path)
torch.save(model.state_dict(), "model.pth")
results_df.to_csv("evaluation_results.csv", index=False)
print("CSV 文件已保存到 evaluation_results.csv")
