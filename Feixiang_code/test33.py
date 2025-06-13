import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
import matplotlib.pyplot as plt

# 设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# 数据集定义
class MotionDataset(Dataset):
    def __init__(self, files_lst, attr):
        self.files = files_lst
        self.features = attr

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data_input = pd.read_csv(self.files[idx] + "trajectory.csv").iloc[:, 1:].values  # [T, 9*N]
        data_output = pd.read_csv(self.files[idx] + "model.csv").loc[:, self.features].values  # [T, D]

        T = data_input.shape[0]
        pc = torch.tensor(data_input, dtype=torch.float32).reshape(T, 9, -1)  # 直接 reshape 一次性搞定
        tensor_output = torch.tensor(data_output, dtype=torch.float32)  # [T,D]

        return pc, tensor_output


def collate_fn(batch):
    seqs, labs = zip(*batch)
    seqs_padded = pad_sequence(seqs, batch_first=True, padding_value=0)  # [B,T,9,N]
    labs_padded = pad_sequence(labs, batch_first=True, padding_value=0)  # [B,T,D]
    lengths = torch.tensor([s.shape[0] for s in seqs], dtype=torch.long)
    return seqs_padded, labs_padded, lengths


# 文件列表 & 特征
# path = "D:/dataset/COMP5703/datasetX/test"
# os.chdir(path)
# files = [fn[:-9] for fn in os.listdir(path) if fn.endswith("model.csv")]
feature_list = ["KneeMoment"]
features = [f"{element}_{axis}" for element in feature_list for axis in ["Y"]]
# test_dataset = MotionDataset(files, features)
# test_loader = DataLoader(test_dataset,  batch_size=4, shuffle=False, collate_fn=collate_fn)
# # n = len(dataset)
#
# path = "D:/dataset/COMP5703/datasetX/train"
# os.chdir(path)
# files_train = [fn[:-9] for fn in os.listdir(path) if fn.endswith("model.csv")]
# train_dataset = MotionDataset(files_train, features)
# train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True,  collate_fn=collate_fn)
#
# path = "D:/dataset/COMP5703/datasetX/val"
# os.chdir(path)
# files_val = [fn[:-9] for fn in os.listdir(path) if fn.endswith("model.csv")]
# val_dataset = MotionDataset(files_val, features)
# val_loader   = DataLoader(val_dataset,   batch_size=4, shuffle=False, collate_fn=collate_fn)
#
path = "D:/dataset/COMP5703/datasetF"
os.chdir(path)
# 在最开始定义好各路数据的目录
path_test = "D:/dataset/COMP5703/datasetF/test"
path_train = "D:/dataset/COMP5703/datasetF/train"
path_val = "D:/dataset/COMP5703/datasetF/val"


def list_prefixes(folder):
    # 找到所有 *_model.csv，去掉后缀，保留完整路径前缀
    files = [fn for fn in os.listdir(folder) if fn.endswith("model.csv")]
    return [os.path.join(folder, fn[:-9]) for fn in files]


# 分别构造三份列表
files_test = list_prefixes(path_test)
files_train = list_prefixes(path_train)
files_val = list_prefixes(path_val)

# Dataset 时直接传入带路径的前缀列表；不要 os.chdir
test_dataset = MotionDataset(files_test, features)
train_dataset = MotionDataset(files_train, features)
val_dataset = MotionDataset(files_val, features)

batch_size = 16

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, pin_memory=True)



####################################
#   定义 PointNetEncoder、BiLSTM   #
####################################
class STN3d(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.fc = nn.Linear(channel, channel)

    def forward(self, x):
        B, C, N = x.shape
        return torch.eye(C, device=x.device).unsqueeze(0).repeat(B, 1, 1)


class PointNetEncoder(nn.Module):
    def __init__(self, global_feat=True, channel=9):
        super().__init__()
        self.stn = STN3d(channel)
        self.conv1 = nn.Conv1d(channel, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.global_feat = global_feat

    def forward(self, x):
        B, D, N = x.size()
        trans = self.stn(x)
        x = x.transpose(2, 1).bmm(trans).transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = torch.max(x, 2, keepdim=True)[0].view(B, -1)  # [B,128]
        return x, trans, None


class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x, lengths):
        # x: [B, T, input_size]
        # lengths: [B]
        packed = pack_padded_sequence(x,
                                      lengths.cpu(),
                                      batch_first=True,
                                      enforce_sorted=False)  # ← returns one object now
        packed_output, _ = self.lstm(packed)  # feed the PackedSequence into LSTM
        output, _ = pad_packed_sequence(packed_output,
                                        batch_first=True)  # unpack back to tensor [B, T, hidden*2]
        return self.fc(output)


class FullModel(nn.Module):
    def __init__(self, pointnet, lstm):
        super().__init__()
        self.pointnet = pointnet
        self.lstm = lstm

    def forward(self, x, lengths):
        B, T, C, N = x.shape
        x2 = x.view(B * T, C, N)
        feat, _, _ = self.pointnet(x2)  # [B*T,128]
        seq_feat = feat.view(B, T, -1)  # [B,T,128]
        return self.lstm(seq_feat, lengths)  # [B,T,D]


# 实例化并加载预训练（如有）
pointnet = PointNetEncoder(channel=9).to(device)
bilstm = BiLSTM(input_size=128, hidden_size=96, num_layers=3, output_size=len(features)).to(device)
if os.path.exists("pointnet.pth"):
    ck = torch.load("pointnet.pth", map_location=device)
    pointnet.load_state_dict(ck["model_state_dict"])
if os.path.exists("bilstm.pth"):
    ck = torch.load("bilstm.pth", map_location=device)
    bilstm.load_state_dict(ck["model_state_dict"])
model = FullModel(pointnet, bilstm).to(device)


####################################
#       自定义加权MSE损失函数      #
####################################
# def weighted_mse_loss(outputs, labels, lengths, w_front=4.0, w_back=4.0):
#     B, T_max, D = outputs.shape
#     losses = []
#     for b in range(B):
#         L = lengths[b].item()
#         pred = outputs[b, :L, :]  # [L,D]
#         truth = labels[b, :L, :]
#         se = (pred - truth).pow(2)  # [L,D]
#
#         half = L // 2
#         true_1d = truth[:, 0]
#         idx_f = int(true_1d[:half].argmax().item())
#         idx_b = half + int(true_1d[half:].argmax().item())
#
#         weights = torch.ones(L, device=outputs.device)
#         weights[idx_f] = w_front
#         weights[idx_b] = w_back
#
#         # 加权后再求均值
#         loss_b = (weights.unsqueeze(1) * se).sum() / (weights.sum() * D)
#         losses.append(loss_b)
#     return torch.stack(losses).mean()
criterion = nn.MSELoss().to(device)

####################################
#        训练／验证／测试环节       #
####################################

optimizer = optim.Adam(model.parameters(), lr=1e-3)
epochs = 50

for epoch in range(1, epochs + 1):
    # --- 训练 ---
    model.train()
    running_loss = 0.0
    for x, y, lengths in train_loader:
        x, y, lengths = x.to(device), y.to(device), lengths.to(device)
        optimizer.zero_grad()
        out = model(x, lengths)  # [B,T,D]
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    train_loss = running_loss / len(train_loader)

    # --- 验证 ---
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for x, y, lengths in val_loader:
            x, y, lengths = x.to(device), y.to(device), lengths.to(device)
            out = model(x, lengths)
            val_loss += criterion(out, y).item()
    val_loss /= len(val_loader)

    print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
# --- 测试集评估 ---
model.eval()
test_loss = 0.0
with torch.no_grad():
    for x, y, lengths in test_loader:
        x, y, lengths = x.to(device), y.to(device), lengths.to(device)
        out = model(x, lengths)  # [B, T, D]
        test_loss += criterion(out, y).item()
test_loss /= len(test_loader)
print(f"Test Loss: {test_loss:.4f}")

# --- 测试 & 可视化略同原代码，不再赘述 ---
model.eval()

results = []  # 存放每个样本的 [f_true, f_pred, b_true, b_pred]
with torch.no_grad():
    for inputs, labels, lengths in test_loader:
        inputs, labels, lengths = inputs.to(device), labels.to(device), lengths.to(device)
        outputs = model(inputs, lengths)  # [B, T_max, D]

        B = inputs.size(0)
        for i in range(B):
            L = lengths[i].item()
            true_seq = labels[i, :L, 0].cpu().numpy()  # 取第 0 维
            pred_seq = outputs[i, :L, 0].cpu().numpy()

            half = L // 2
            # 前半段
            idx_f = np.argmax(true_seq[:half])
            f_true = true_seq[idx_f]
            f_pred = pred_seq[idx_f]
            # 后半段
            idx_b = half + np.argmax(true_seq[half:])
            b_true = true_seq[idx_b]
            b_pred = pred_seq[idx_b]

            results.append([f_true, f_pred, b_true, b_pred])

# model.eval()
# with torch.no_grad():
#     # 拿一个 batch 中第一个样本
#     for batch in test_loader:
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
# # 转为 DataFrame 并打印
# path = "D:/dataset/COMP5703"
# os.chdir(path)
# df = pd.DataFrame(results,
#                   columns=[
#                       "front_true_max",
#
#                       "front_pred_at_true_max",
#                       "back_true_max",
#                       "back_pred_at_true_max"
#                   ])
# df.to_csv("hahaha0.csv")

# --- 测试 & 收集结果（不改动） ---
model.eval()

results = []  # 存放每个样本的 [f_true, f_pred, b_true, b_pred]
with torch.no_grad():
    for inputs, labels, lengths in test_loader:
        inputs, labels, lengths = inputs.to(device), labels.to(device), lengths.to(device)
        outputs = model(inputs, lengths)  # [B, T_max, D]
        B = inputs.size(0)
        for i in range(B):
            L = lengths[i].item()
            true_seq = labels[i, :L, 0].cpu().numpy()
            pred_seq = outputs[i, :L, 0].cpu().numpy()

            half = L // 2
            idx_f = np.argmax(true_seq[:half])
            f_true = true_seq[idx_f]
            f_pred = pred_seq[idx_f]

            idx_b = half + np.argmax(true_seq[half:])
            b_true = true_seq[idx_b]
            b_pred = pred_seq[idx_b]

            results.append([f_true, f_pred, b_true, b_pred])

# --- 这里开始修改：把原始文件名对应上去 ---
# test_dataset 是 Subset，直接拿它的 indices 属性：
# file_indices = test_dataset.indices  # e.g. [23, 7, 5, 42, ...]
# # 用它去索引原始的 files 列表，得到测试集里每个样本的文件名前缀
# test_file_names = [files_test[idx] for idx in file_indices]
#
# # 构建 DataFrame
# df = pd.DataFrame(
#     results,
#     columns=[
#         "front_true_max",
#         "front_pred_at_true_max",
#         "back_true_max",
#         "back_pred_at_true_max"
#     ]
# )
# # 插入一列 file_name
# df["file_name"] = test_file_names
#
# # 保存到 CSV
# df.to_csv("hahaha0_with_filenames.csv", index=False)
# print("Saved results to hahaha0_with_filenames.csv")
prefix = "D:/dataset/COMP5703/datasetF/test/"
prefix_len = len(prefix)

# 收集完 results 之后
# 切片去掉前缀
test_file_names = [p[prefix_len:] for p in files_test]

# 构建 DataFrame
df = pd.DataFrame(
    results,
    columns=[
        "front_true_max",
        "front_pred_at_true_max",
        "back_true_max",
        "back_pred_at_true_max"
    ]
)
# 插入一列 file_name
df["file_name"] = test_file_names

# 保存到 CSV
df.to_csv("hahaha00_with_filenames.csv", index=False)
print("Saved results to hahaha00_with_filenames.csv")
