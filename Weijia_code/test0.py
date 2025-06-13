import os
import glob
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ========== 数据加载函数 ==========

EXCLUDED_POINTS = ['LFHD', 'RFHD', 'LBHD', 'RBHD', 'C7', 'T10', 'CLAV', 'RBAK']
NUM_FEATURES = 9
FIXED_POINTS = ['LANK', 'LHEE', 'LKNE', 'LTOE', 'LTHI', 'LTIAD', 'LTIAP', 'LTIB', 'LKNM', 'LTHAD', 'LTHAP', 'LPSI',
                'LASI', 'LSMH', 'LFMH', 'LVMH', 'LMED',
                'RANK', 'RHEE', 'RKNE', 'RTOE', 'RTHI', 'RTIAD', 'RTIAP', 'RTIB', 'RKNM', 'RTHAD', 'RTHAP', 'RPSI',
                'RASI', 'RSMH', 'RFMH', 'RVMH', 'RMED']


def is_left_file(filename):
    return os.path.basename(filename).startswith("L")


def pair_files(folder_path):
    traj_files = sorted(glob.glob(os.path.join(folder_path, '*_trajectory.csv')))
    model_files = sorted(glob.glob(os.path.join(folder_path, '*_model.csv')))
    pairs = []
    for traj in traj_files:
        base = os.path.basename(traj).replace('_trajectory.csv', '')
        model = os.path.join(folder_path, base + '_model.csv')
        if os.path.exists(model):
            pairs.append((traj, model))
    return pairs


def read_trajectory(traj_path, is_left):
    df = pd.read_csv(traj_path)
    suffixes = ['X', 'Y', 'Z', "X'", "Y'", "Z'", "X''", "Y''", "Z''"]
    prefix = 'L' if is_left else 'R'
    point_list = [p for p in FIXED_POINTS if p.startswith(prefix)]
    cols = [f"{p}_{s}" for p in point_list for s in suffixes if f"{p}_{s}" in df.columns]
    X = df[cols].values.reshape(-1, len(point_list), NUM_FEATURES)
    return X


def read_kam(model_path, is_left):
    df = pd.read_csv(model_path)
    col = 'LKneeMoment_Y' if is_left else 'RKneeMoment_Y'
    return df[col].values


# ========== Dataset 类 ==========

class KAMDataset(Dataset):
    def __init__(self, folder_path):
        self.pairs = pair_files(folder_path)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        traj_path, model_path = self.pairs[idx]
        is_left = is_left_file(traj_path)
        X = read_trajectory(traj_path, is_left)
        Y = read_kam(model_path, is_left)
        filename = os.path.basename(model_path)
        return {
            'input': torch.tensor(X, dtype=torch.float32),
            'target': torch.tensor(Y, dtype=torch.float32),
            'filename': filename
        }


# ========== collate_fn ==========

def collate_fn(batch):
    inputs = [item['input'] for item in batch]
    targets = [item['target'] for item in batch]
    lengths = [x.shape[0] for x in inputs]
    max_len = max(lengths)

    padded_inputs = pad_sequence(inputs, batch_first=True)  # (B, T, N, D)
    padded_targets = pad_sequence([y.unsqueeze(-1) for y in targets], batch_first=True).squeeze(-1)  # (B, T)

    mask = torch.zeros((len(batch), max_len), dtype=torch.bool)
    for i, l in enumerate(lengths):
        mask[i, :l] = 1

    return padded_inputs, padded_targets, mask


# ========== 模型结构 ==========

import torch
import torch.nn as nn
import torch.nn.functional as F


class PointNetFrameEncoder(nn.Module):
    def __init__(self, input_dim=9, hidden_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(), nn.BatchNorm1d(64),
            nn.Linear(64, 128), nn.ReLU(), nn.BatchNorm1d(128),
            nn.Linear(128, hidden_dim), nn.ReLU(), nn.BatchNorm1d(hidden_dim),
        )

    def forward(self, x):  # x: (B, T, N, D)
        B, T, N, D = x.shape
        x = x.reshape(B * T * N, D)
        x = self.mlp(x)
        x = x.view(B * T, N, -1)
        x = torch.max(x, dim=1)[0]  # (B*T, F)
        return x.view(B, T, -1)  # (B, T, F)


class PointNetTransformerRegressor(nn.Module):
    def __init__(self, input_dim=9, point_feat_dim=256, trans_hidden=256, nhead=8, num_layers=3):
        super().__init__()
        self.encoder = PointNetFrameEncoder(input_dim, point_feat_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=point_feat_dim,
            nhead=nhead,
            dim_feedforward=trans_hidden,
            dropout=0.2,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.head = nn.Sequential(
            nn.Linear(point_feat_dim, 128), nn.ReLU(), nn.BatchNorm1d(128), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(), nn.BatchNorm1d(64), nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x, mask):
        """
        x: (B, T, N, D)
        mask: (B, T) with bool values: True for valid, False for pad
        """
        feat = self.encoder(x)  # (B, T, F)
        key_padding_mask = ~mask  # (B, T), True for padding, False for valid

        # transformer expects key_padding_mask: True means **ignore**
        trans_out = self.transformer(feat, src_key_padding_mask=key_padding_mask)  # (B, T, F)

        B, T, F = trans_out.shape
        out = self.head(trans_out.reshape(B * T, F)).reshape(B, T)  # (B, T)

        return out * mask  # apply mask to ensure padding has no prediction


# ========== 损失函数 ==========

def masked_mse_loss(pred, target, mask):
    loss = F.mse_loss(pred, target, reduction='none')
    return (loss * mask).sum() / mask.sum()


# ========== 训练函数 ==========

def train_model(model, train_loader, val_loader, epochs=500, lr=1e-3, patience=50, save_path='best_model.pth'):
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # best_loss = float('inf')
    # patience_counter = 0
    # train_losses, val_losses = [], []
    #
    # for epoch in range(epochs):
    #     model.train()
    #     total_train_loss = 0
    #     for x, y, m in train_loader:
    #         pred = model(x, m)
    #         loss = masked_mse_loss(pred, y, m)
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #         total_train_loss += loss.item()
    #
    #     model.eval()
    #     total_val_loss = 0
    #     with torch.no_grad():
    #         for x, y, m in val_loader:
    #             pred = model(x, m)
    #             loss = masked_mse_loss(pred, y, m)
    #             total_val_loss += loss.item()
    model = model.to(device)  # 将模型移动到GPU
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_loss = float('inf')
    patience_counter = 0
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        for x, y, m in train_loader:
            # 将数据移动到GPU
            x, y, m = x.to(device), y.to(device), m.to(device)

            optimizer.zero_grad()
            pred = model(x, m)
            loss = masked_mse_loss(pred, y, m)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for x, y, m in val_loader:
                x, y, m = x.to(device), y.to(device), m.to(device)
                pred = model(x, m)
                loss = masked_mse_loss(pred, y, m)
                total_val_loss += loss.item()

        train_loss = total_train_loss / len(train_loader)
        val_loss = total_val_loss / len(val_loader)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), save_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curve')
    plt.savefig('loss_curve.png')
    plt.close()


# ========== 评估函数 ==========

def evaluate_model(model, test_loader, save_dir='results2'):
    model = model.to(device)  # 确保模型在GPU上
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    total_mse, total_frames = 0, 0

    with torch.no_grad():
        for i, (x, y, m) in enumerate(test_loader):
            # 将数据移动到GPU
            x, y, m = x.to(device), y.to(device), m.to(device)

            pred = model(x, m)
            # 将数据移回CPU进行后续处理
            pred_cpu = pred.cpu()
            y_cpu = y.cpu()
            m_cpu = m.cpu()

            for b in range(x.size(0)):
                T = int(m_cpu[b].sum().item())
                gt = y_cpu[b][:T].numpy()
                pr = pred_cpu[b][:T].numpy()
                mse = np.mean((gt - pr) ** 2)
                total_mse += mse * T
                total_frames += T

                plt.figure()
                plt.plot(gt, label='GT')
                plt.plot(pr, label='Pred')
                plt.xlabel('Frame')
                plt.ylabel('KAM')
                plt.title(f'Sample {i * test_loader.batch_size + b}, MSE: {mse:.4f}')
                plt.legend()
                plt.savefig(os.path.join(save_dir, f'sample_{i * test_loader.batch_size + b}.png'))
                plt.close()

    avg_mse = total_mse / total_frames
    print(f"Test MSE: {avg_mse:.6f}")


# folder = 'D:/dataset/COMP5703/dataset'
# dataset = KAMDataset(folder)
# print(f"找到样本对数量: {len(dataset)}")
#
# total_len = len(dataset)
# train_len = int(0.8 * total_len)
# val_len = int(0.1 * total_len)
# test_len = total_len - train_len - val_len
# train_set, val_set, test_set = random_split(dataset, [train_len, val_len, test_len])

train_set = KAMDataset('D:/dataset/COMP5703/datasetX/train')
test_set = KAMDataset('D:/dataset/COMP5703/datasetX/test')
val_set = KAMDataset('D:/dataset/COMP5703/datasetX/val')

train_loader = DataLoader(train_set, batch_size=64, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_set, batch_size=64, collate_fn=collate_fn)
test_loader = DataLoader(test_set, batch_size=16, collate_fn=collate_fn)

model = PointNetTransformerRegressor().to(device)
train_model(model, train_loader, val_loader, epochs=50)
model.load_state_dict(torch.load('best_model.pth'))
evaluate_model(model, test_loader)
print("✅ 训练和测试可视化完成")


def plot_prediction_vs_ground_truth(model, dataset, sample_idx=0, device='cpu', save_path=None, show=True):
    model = model.to(device)
    model.eval()

    sample = dataset[sample_idx]
    # 将数据移动到GPU
    x = sample['input'].unsqueeze(0).to(device)
    y = sample['target'].unsqueeze(0).to(device)
    T = x.shape[1]
    mask = torch.ones((1, T), dtype=torch.bool).to(device)

    with torch.no_grad():
        pred = model(x, mask)

    # 将数据移回CPU用于绘图
    true_kam = y.squeeze(0).cpu().numpy()
    pred_kam = pred.squeeze(0).cpu().numpy()

    plt.figure(figsize=(10, 5))
    plt.plot(true_kam, label='Ground Truth', linewidth=2)
    plt.plot(pred_kam, label='Prediction', linestyle='--')
    plt.xlabel('Frame')
    plt.ylabel('KAM')
    plt.title(f'Sample {sample_idx} - KAM Prediction vs Ground Truth')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    elif show:
        plt.show()


plot_prediction_vs_ground_truth(model, test_set, sample_idx=2)


def extract_peak_kam_metrics(model, dataset,
                             save_peak_path='results2/peak_kam_summary.csv',
                             save_frame_path='results2/frame_kam_details.csv',
                             device='cpu'):
    """
    修改后函数同时生成两个文件:
    1. peak_kam_summary.csv - 保留原始峰值统计
    2. frame_kam_details.csv - 新增逐帧数据记录
    """
    model.eval()
    peak_records = []
    frame_records = []

    for idx in range(len(dataset)):
        sample = dataset[idx]
        x = sample['input'].unsqueeze(0).to(device)  # (1, T, N, D)
        y = sample['target'].unsqueeze(0).to(device)  # (1, T)
        T = x.shape[1]
        mask = torch.ones((1, T), dtype=torch.bool).to(device)
        filename = sample['filename']

        with torch.no_grad():
            pred = model(x, mask)  # (1, T)

        # 转换为numpy数组
        pred_kam = pred.squeeze(0).cpu().numpy()
        true_kam = y.squeeze(0).cpu().numpy()

        # ===== 1. 峰值统计部分 =====
        mid = T // 2
        true_kam1 = np.max(true_kam[:mid]) if mid > 0 else np.nan
        true_kam2 = np.max(true_kam[mid:]) if mid < T else np.nan
        pred_kam1 = np.max(pred_kam[:mid]) if mid > 0 else np.nan
        pred_kam2 = np.max(pred_kam[mid:]) if mid < T else np.nan

        peak_records.append({
            'filename': filename,
            'peak KAM1 Actual': true_kam1,
            'peak KAM1 Predict': pred_kam1,
            'peak KAM2 Actual': true_kam2,
            'peak KAM2 Predict': pred_kam2,
        })

        # ===== 2. 逐帧数据部分 =====
        for frame_idx in range(T):
            frame_records.append({
                'filename': filename,
                'frame': frame_idx,
                'actual_kam': true_kam[frame_idx],
                'predicted_kam': pred_kam[frame_idx]
            })

    # 保存峰值统计
    peak_df = pd.DataFrame(peak_records)
    peak_df.to_csv(save_peak_path, index=False)
    print(f"✅ 峰值KAM统计已保存至: {save_peak_path}")

    # 保存逐帧数据
    frame_df = pd.DataFrame(frame_records)
    frame_df.to_csv(save_frame_path, index=False)
    print(f"✅ 逐帧KAM数据已保存至: {save_frame_path}")

    return peak_df, frame_df


# 运行修改后的函数
peak_df, frame_df = extract_peak_kam_metrics(
    model,
    test_set,
    save_peak_path='results2/peak_summary.csv',
    save_frame_path='results2/frame_details.csv',
)

# from numpy import mean, std
#
# df = extract_peak_kam_metrics(model, test_set, device='cpu')
#
# kam1_mse = mean((df['peak KAM1 Actual'] - df['peak KAM1 Predict'])**2)
# kam2_mse = mean((df['peak KAM2 Actual'] - df['peak KAM2 Predict'])**2)
# kam1_std = std((df['peak KAM1 Actual'] - df['peak KAM1 Predict'])**2)
# kam2_std = std((df['peak KAM2 Actual'] - df['peak KAM2 Predict'])**2)
#
# print("📊 外部单独统计结果：")
# print(f"Peak KAM1 MSE  → Mean = {kam1_mse:.6f}")
# print(f"Peak KAM2 MSE  → Mean = {kam2_mse:.6f}")