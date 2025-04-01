import os
import pandas as pd

# 设置源文件夹和目标文件夹路径
source_folder = "D:/dataset/COMP5703/dataset"  # 根据实际情况修改
target_folder = "D:/dataset/COMP5703/dataset3"
os.makedirs(target_folder, exist_ok=True)

# 获取所有以 _trajectory.csv 结尾的文件
files = [f for f in os.listdir(source_folder) if f.endswith("_model.csv")]
if not files:
    print("没有找到任何以 _trajectory.csv 结尾的文件")
    exit()

# 保存每个文件处理后的DataFrame和对应的特征集合
processed_dfs = {}
features_list = []

def process_columns(df, prefix):
    """
    根据文件前缀处理DataFrame的列：
      - 如果 prefix == 'L': 只保留以L开头的（去掉L）和非R开头的列；
      - 如果 prefix == 'R': 只保留以R开头的（去掉R）和非L开头的列；
      - 非L非R开头的列保持不变。
    """
    new_columns = {}
    for col in df.columns:
        # 判断列开头字符
        if col.startswith('L') or col.startswith('R'):
            if prefix == 'L':
                # 如果文件以L开头，则舍弃所有以R开头的特征
                if col.startswith('R'):
                    continue
                elif col.startswith('L'):
                    # 去掉开头的 "L"
                    new_name = col[1:]
                    new_columns[col] = new_name
            elif prefix == 'R':
                # 如果文件以R开头，则舍弃所有以L开头的特征
                if col.startswith('L'):
                    continue
                elif col.startswith('R'):
                    # 去掉开头的 "R"
                    new_name = col[1:]
                    new_columns[col] = new_name
        else:
            # 非L非R开头的列保持不变
            new_columns[col] = col

    # 仅保留 new_columns 中指定的列，并同时重命名列
    df_processed = df.loc[:, list(new_columns.keys())].rename(columns=new_columns)
    return df_processed

# 先处理每个文件，并记录处理后各文件的特征集合
for file in files:
    file_path = os.path.join(source_folder, file)
    try:
        df = pd.read_csv(file_path)
        # 根据文件名的首字母判断处理规则（假设文件名第一个字符为 L 或 R）
        file_prefix = file[0]
        if file_prefix not in ['L', 'R']:
            print(f"文件 {file} 的前缀不为L或R，跳过处理。")
            continue

        df_processed = process_columns(df, file_prefix)
        processed_dfs[file] = df_processed
        features_list.append(set(df_processed.columns))
    except Exception as e:
        print(f"处理文件 {file} 时出错: {e}")

if not features_list:
    print("没有处理到任何文件。")
    exit()

# 计算所有文件的共有特征（交集）
common_features = set.intersection(*features_list)
print("所有文件共有的特征:")
print(common_features)

# 遍历处理后的文件，仅保留共有的特征，并保存到目标文件夹
for file, df_processed in processed_dfs.items():
    # 保留共有特征
    df_common = df_processed.loc[:, df_processed.columns.intersection(common_features)]
    target_file_path = os.path.join(target_folder, file)
    df_common.to_csv(target_file_path, index=False)
    print(f"保存文件: {target_file_path}")
