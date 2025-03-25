import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split


# 定义位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=0.1)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # 变为 [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# 定义多头注意力
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        attn_output, _ = self.attention(query, key, value, key_padding_mask=mask)
        return self.norm(query + self.dropout(attn_output))


# 定义Encoder层
class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_output = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        return x


# 定义Decoder层
class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.attention1 = MultiHeadAttention(d_model, n_heads, dropout)
        self.attention2 = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, mask=None, memory_mask=None):
        attn_output = self.attention1(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        attn_output2 = self.attention2(x, encoder_output, encoder_output, memory_mask)
        x = self.norm2(x + self.dropout(attn_output2))
        ffn_output = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_output))
        return x


# 定义整个模型
class Informer(nn.Module):
    def __init__(self, input_size, d_model, n_heads, num_layers, output_size, seq_len, label_len, out_len, dropout=0.1):
        super(Informer, self).__init__()
        self.seq_len = seq_len
        self.label_len = label_len
        self.out_len = out_len
        self.d_model = d_model

        # 输入全连接层
        self.input_fc = nn.Linear(input_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)

        # Encoder部分
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, n_heads, dropout) for _ in range(num_layers)])

        # Decoder部分
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, n_heads, dropout) for _ in range(num_layers)])

        # 输出层
        self.fc_out = nn.Linear(d_model, output_size)

    def forward(self, x, encoder_mask=None, decoder_mask=None):
        # x的形状是 [batch_size, seq_len, input_size]
        x = self.input_fc(x)  # [batch_size, seq_len, d_model]
        x = self.pos_encoding(x)  # 加上位置编码

        # 经过Encoder部分
        encoder_output = x
        for encoder_layer in self.encoder_layers:
            encoder_output = encoder_layer(encoder_output, encoder_mask)

        # 经过Decoder部分
        decoder_input = encoder_output
        for decoder_layer in self.decoder_layers:
            decoder_input = decoder_layer(decoder_input, encoder_output, decoder_mask)

        # 获取最终的预测输出
        output = self.fc_out(decoder_input)
        return output


# 数据集类
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


# 定义 collate_fn 函数
def collate_fn(batch):
    inputs, labels = zip(*batch)
    inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=0)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=0)
    input_lengths = torch.tensor([x.shape[0] for x in inputs])
    return inputs_padded, labels_padded, input_lengths


# 数据读取函数
def read_csv(filepath, feature):
    return pd.read_csv(filepath).loc[:, feature].values


# 设备设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 设置数据路径
path = "D:/dataset/COMP5703/dataset"
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

# 特征选择
feature_list = ["LHipAngles"]
features = [f"{element}_{axis}" for element in feature_list for axis in ["X", "Y", "Z"]]

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

# 初始化模型
input_size = pd.read_csv(files[0] + "trajectory.csv").shape[1] - 1
d_model = 512  # Transformer 中的特征维度
nhead = 8
num_layers = 2
dropout = 0.2
learning_rate = 0.00005
output_size = len(features)

model = Informer(input_size=input_size, d_model=d_model, n_heads=nhead, num_layers=num_layers,
                 output_size=output_size, seq_len=100, label_len=100, out_len=50, dropout=dropout).to(device)

criterion = nn.MSELoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练过程
epochs = 50
for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    for batch in train_dataloader:
        inputs, labels, lengths = batch
        inputs, labels, lengths = inputs.to(device), labels.to(device), lengths.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / len(train_dataloader):.4f}")

# 测试过程
model.eval()
total_test_loss = 0.0
with torch.no_grad():
    for batch in test_dataloader:
        inputs, labels, lengths = batch
        inputs, labels, lengths = inputs.to(device), labels.to(device), lengths.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        total_test_loss += loss.item()

print(f"Test Loss: {total_test_loss / len(test_dataloader):.4f}")
