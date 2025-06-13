import json
import os
import shutil

input_dir = 'D:/dataset/COMP5703/dataset'
output_dir = 'D:/dataset/COMP5703/datasetX'

# 加载 ID 并复制函数
def load_ids_and_copy(split_name):
    os.chdir('D:/dataset/COMP5703/dataset')
    with open(f"{split_name}_ids.json", "r") as f:
        base_ids = json.load(f)

    split_dir = os.path.join(output_dir, split_name)
    os.makedirs(split_dir, exist_ok=True)

    for base in base_ids:
        model_file = f"{base}_model.csv"
        traj_file = f"{base}_trajectory.csv"

        model_src = os.path.join(input_dir, model_file)
        traj_src = os.path.join(input_dir, traj_file)

        model_dst = os.path.join(split_dir, model_file)
        traj_dst = os.path.join(split_dir, traj_file)

        if os.path.exists(model_src) and os.path.exists(traj_src):
            shutil.copy(model_src, model_dst)
            shutil.copy(traj_src, traj_dst)
        else:
            print(f"⚠️ 缺少文件: {base}")

# 执行复制
for split in ['train', 'val', 'test']:
    load_ids_and_copy(split)

print("✅ 文件复制完成，按已保存 ID 恢复划分结构。")
