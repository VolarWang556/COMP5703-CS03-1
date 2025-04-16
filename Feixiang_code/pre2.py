import os
import pandas as pd

# 设置源文件夹和目标文件夹路径
source_folder = "D:/dataset/COMP5703/dataset"
target_folder = "D:/dataset/COMP5703/dataset4"
os.makedirs(target_folder, exist_ok=True)

# 要处理的文件类型
file_suffixes = ["_model.csv", "_trajectory.csv"]

# 定义 B 的顺序
B_order = ["X", "Y", "Z", "X'", "Y'", "Z'", "X''", "Y''", "Z''"]

# 排序函数（不包括第一列）
def sort_columns(columns):
    def column_key(col):
        if "_" not in col:
            return (col, -1)
        a, b = col.split("_", 1)
        b_index = B_order.index(b) if b in B_order else len(B_order)
        return (a, b_index)
    return sorted(columns, key=column_key)

# 处理列名
def process_columns(df, prefix):
    new_columns = {}
    for col in df.columns:
        if col.startswith('L') or col.startswith('R'):
            if prefix == 'L':
                if col.startswith('R'):
                    continue
                elif col.startswith('L'):
                    new_name = col[1:]
                    new_columns[col] = new_name
            elif prefix == 'R':
                if col.startswith('L'):
                    continue
                elif col.startswith('R'):
                    new_name = col[1:]
                    new_columns[col] = new_name
        else:
            new_columns[col] = col
    df_processed = df.loc[:, list(new_columns.keys())].rename(columns=new_columns)
    return df_processed

# 对每种类型的文件分别处理
for suffix in file_suffixes:
    print(f"\n处理文件类型: {suffix}")
    files = [f for f in os.listdir(source_folder) if f.endswith(suffix)]
    if not files:
        print(f"没有找到任何以 {suffix} 结尾的文件")
        continue

    processed_dfs = {}
    features_list = []

    for file in files:
        file_path = os.path.join(source_folder, file)
        try:
            df = pd.read_csv(file_path)
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
        print(f"{suffix} 类型的文件没有成功处理。")
        continue

    common_features = set.intersection(*features_list)
    print(f"{suffix} 类型的所有文件共有特征:")
    print(common_features)

    for file, df_processed in processed_dfs.items():
        # 拆分出第一列和其余列
        first_col_name = df_processed.columns[0]
        remaining_cols = df_processed.columns[1:]

        # 保留交集特征（不包含第一列）
        common_feature_cols = [col for col in remaining_cols if col in common_features]

        # 排序特征列
        sorted_features = sort_columns(common_feature_cols)

        # 最终列顺序：第一列 + 排好序的特征列
        final_columns = [first_col_name] + sorted_features
        df_sorted = df_processed[final_columns]

        # 保存
        target_file_path = os.path.join(target_folder, file)
        df_sorted.to_csv(target_file_path, index=False)
        print(f"保存文件: {target_file_path}")


