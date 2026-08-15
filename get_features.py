import torch
import os
import numpy as np
import pandas as pd
import torch.nn.functional as F
import torch


def check_feature_dimensions(pt_directory):
    for filename in os.listdir(pt_directory):
        if filename.endswith('.pt'):
            # 加载 .pt 文件
            filepath = os.path.join(pt_directory, filename)
            data = torch.load(filepath)

            # 输出特征的维度
            if "representations" in data:
                for layer, reps in data["representations"].items():
                    print(f"File: {filename}, Layer: {layer}, Dimension: {reps.shape}")
            else:
                print(f"File: {filename} does not contain 'representations'.")

def load_pt_features(pt_filename):
    """加载 .pt 文件中的全局特征，确保提取正确的数据"""
    try:
        data = torch.load(pt_filename)
        # 检查是否存在 'mean_representations' 并提取全局特征
        if "mean_representations" in data:
            global_features = data["mean_representations"][0].numpy()  # 提取第一个全局特征
            return global_features
        else:
            print(f"Warning: 'mean_representations' not found in {pt_filename}.")
            return None
    except Exception as e:
        print(f"Error loading file {pt_filename}: {e}")
        return None


def load_dat_matrix(dat_filename):
    """从 .dat 文件加载特征矩阵并展平"""
    try:
        # 使用 NumPy 从文本文件加载数据
        data = np.loadtxt(dat_filename)  # 默认以空格分隔
        return torch.flatten(torch.tensor(data, dtype=torch.float32))  # 转换为 1D PyTorch 张量
    except Exception as e:
        print(f"Error loading {dat_filename}: {e}")
        return None

def pool_matrix(matrix, target_size=(1, 1)):
    """对输入矩阵进行池化操作，返回固定大小的特征向量."""
    # 确保输入是4D (N, C, H, W), 扩展维度
    matrix = matrix.unsqueeze(0).unsqueeze(0)  # 添加 batch 维度和 channel 维度

    pooled_matrix = F.adaptive_avg_pool2d(matrix, target_size)  # 使用平均池化
    return pooled_matrix.squeeze(0).squeeze(0)  # 移除多余的维度

def merge_features(protein_name, pt_directory, csv_file, dat_directory):
    """合并特征"""

    # 1. 加载 320 维特征
    pt_filename = os.path.join(pt_directory, f"{protein_name}.pt")
    pt_features = load_pt_features(pt_filename)

    if pt_features is None:
        print(f"Error loading PT features for {protein_name}.")
        return None

    # 2. 加载 41 个特征
    df = pd.read_csv(csv_file)
    row = df[df['protein_name'] == protein_name]

    if row.empty:
        print(f"Warning: {protein_name} not found in CSV file.")
        return None

    # 提取41个特征并转换为浮点类型
    other_features = row.iloc[0, 1:-1].values  # 从第二列开始取特征
    # 检查数据类型并转换为浮点类型
    other_features = [float(x) for x in other_features]  # 将所有特征值转换为 float
    other_features = torch.tensor(other_features, dtype=torch.float32)  # 转换为 PyTorch 张量

    # 3. 加载 L*L 特征矩阵
    dat_filename = os.path.join(dat_directory, f"{protein_name}.dat")
    l_matrix_features = load_dat_matrix(dat_filename)
    if l_matrix_features is None:
        print(f"Error loading features from {dat_filename}.")
        return None

    pooled_features = pool_matrix(torch.tensor(l_matrix_features, dtype=torch.float32))

    # 确保将所有特征转换为张量
    if isinstance(pt_features, np.ndarray):
        pt_features = torch.tensor(pt_features, dtype=torch.float32)
    if isinstance(pooled_features, np.ndarray):
        pooled_features = torch.tensor(pooled_features, dtype=torch.float32)

    # 4. 合并所有特征
    try:
        # 假设在这里我们得到了多个特征，pt_features 和 l_matrix_features 可能具有不同的尺寸

        # ESM Embedding + experts features
        combined_features = torch.cat((pt_features, other_features, pooled_features.view(-1)))

        # experts features
        # combined_features = torch.cat((other_features, pooled_features.view(-1)))

        # ESM Embedding
        # combined_features = pt_features
    except RuntimeError as e:
        print(f"Error combining features for {protein_name}: {e}")

        # 打印出每个特征的形状
        print(f"pt_features shape: {pt_features.shape}")
        print(f"other_features shape: {other_features.shape}")
        print(f"l_matrix_features shape: {l_matrix_features.shape}")

        return None

    return combined_features

def get_all_features(pt_directory, csv_file, dat_directory):
    protein_names = [f[:-3] for f in os.listdir(pt_directory) if f.endswith('.pt')]
    all_features = []
    all_protein_name = []

    for protein_name in protein_names:
        features = merge_features(protein_name, pt_directory, csv_file, dat_directory)
        if features is not None:
            all_features.append(features)
            all_protein_name.append(protein_name)

    return all_features, all_protein_name