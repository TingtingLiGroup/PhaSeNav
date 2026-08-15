import os
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset, Dataset
from get_features import get_all_features
import pandas as pd
import json
import random
from collections import defaultdict,Counter

class FeaturesDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# baoyq's union features(esm + 41 expert features + 1 matrix data)
def prepare_data(pt_directory, csv_file, dat_directory):
    # 从 get_features.py 获取特征和蛋白质名称
    all_features, all_protein_names = get_all_features(pt_directory, csv_file, dat_directory)

    data = pd.read_csv(csv_file)
    # 提取蛋白质名称和最后一列分数
    csv_protein_names = data.iloc[:, 0].values  # 第一列，蛋白质名称
    scores = data.iloc[:, -1].values  # 最后一列，分数

    # 创建字典以快速查找分数
    score_dict = dict(zip(csv_protein_names, scores))

    # 根据 all_protein_names 的顺序构建标签数组
    labels = []
    for protein_name in all_protein_names:
        if protein_name in score_dict:
            label = 1.0 if score_dict[protein_name] > 0.5 else 0.0  # 根据阈值生成标签
        else:
            label = np.nan  # 若找不到对应的蛋白质名称，可以设为 NaN 或其他默认值
        labels.append(label)


    # 将特征和标签转换为 NumPy 数组
    features_array = np.array([features.numpy() for features in all_features])
    labels_array = np.array(labels)
    label_counts = pd.Series(labels_array).value_counts(dropna=False)

    print(label_counts)

    features_tensor = torch.tensor(features_array, dtype=torch.float32)
    labels_tensor = torch.tensor(labels_array, dtype=torch.float32)

    dataset = FeaturesDataset(features_tensor, labels_tensor)

    # 进行训练集和测试集的划分
    X_train, X_test, y_train, y_test = train_test_split(
        features_array,
        labels_array,
        test_size=0.2,
        stratify=labels_array,  # 需要根据具体标签进行修改
        random_state=42  # 固定种子以保持可重复性
    )

    # 创建 TensorDataset
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                  torch.tensor(y_train))  # 示例标签
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                                 torch.tensor(y_test))  # 示例标签

    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_dataset, test_dataset


def split_train_test_data_int(jsonfile,outputjson):
    # 加载 JSON 文件
    with open(jsonfile, "r") as f:
        proteins = json.load(f)

    # 按类别分组
    category_to_proteins = {}
    for protein in proteins:
        for label, value in protein["labels"].items():
            if value == 1:  # 找到当前蛋白质的类别
                if label not in category_to_proteins:
                    category_to_proteins[label] = []
                category_to_proteins[label].append(protein)
            
    # 按类别划分训练集和测试集
    for label, protein_list in category_to_proteins.items():
        random.shuffle(protein_list)  # 随机打乱
        split_index = int(0.8 * len(protein_list))  # 8:2 划分
        for protein in protein_list[:split_index]:
            protein["split"] = "train"
        for protein in protein_list[split_index:]:
            protein["split"] = "test"

    # 统计每个类别的训练集和测试集样本数
    category_counts = defaultdict(lambda: {"train": 0, "test": 0})
    for protein in proteins:
        for label, value in protein["labels"].items():
            if value == 1:
                category_counts[label][protein["split"]] += 1
                break

    # 打印统计结果
    print("每个类别的训练集和测试集样本数：")
    for label, counts in sorted(category_counts.items(), key=lambda x: int(x[0])):
        print(f"类别 {label}: 训练集={counts['train']}, 测试集={counts['test']}")

    # 保存更新后的 JSON 文件
    with open(outputjson, "w") as f:
        json.dump(proteins, f, indent=4)

    print(f"划分完成，结果已保存到 {outputjson}")


def split_train_test_data_multi_label(jsonfile, outputjson, test_ratio=0.2, seed = 42):

    random.seed(seed)
    # 加载数据
    with open(jsonfile, "r") as f:
        proteins = json.load(f)

    # 过滤掉不属于任何类别的蛋白
    proteins = [protein for protein in proteins if any(value == 1 for value in protein["labels"].values())]
    
    # 初始化类别到蛋白质的映射（一个蛋白可能属于多个类别）
    category_to_proteins = defaultdict(list)
    for protein in proteins:
        for label, value in protein["labels"].items():
            if value == 1:
                category_to_proteins[label].append(protein)
    
    # 为每个蛋白初始化 split 字段（默认为空）
    for protein in proteins:
        protein["split"] = []
    
    # 按类别独立划分
    for label, protein_list in category_to_proteins.items():
        random.shuffle(protein_list)
        split_idx = int(len(protein_list) * (1 - test_ratio))
        
        # 标记训练集和测试集
        for protein in protein_list[:split_idx]:
            if "train" not in protein["split"]:
                protein["split"].append("train")
        for protein in protein_list[split_idx:]:
            if "test" not in protein["split"]:
                protein["split"].append("test")
    
    # 处理冲突：如果一个蛋白被标记为 train 和 test，优先保留 test
    for protein in proteins:
        if "test" in protein["split"]:
            protein["split"] = "test"
        elif "train" in protein["split"]:
            protein["split"] = "train"
        else:
            protein["split"] = "train"  # 默认
    
    # 统计结果
    category_counts = defaultdict(lambda: {"train": 0, "test": 0})
    for protein in proteins:
        for label, value in protein["labels"].items():
            if value == 1:
                category_counts[label][protein["split"]] += 1
    
    print("各类别样本分布：")
    for label, counts in sorted(category_counts.items()):
        print(f"{label}: train={counts['train']}, test={counts['test']}")
    
    # 保存结果
    with open(outputjson, "w") as f:
        json.dump(proteins, f, indent=4)
    print(f"划分完成，结果保存至 {outputjson}")

def split_train_val_test_data_multi_label(jsonfile, outputjson, val_ratio=0.15, test_ratio=0.15, seed=42):
    random.seed(seed)
    
    # 加载数据
    with open(jsonfile, "r") as f:
        proteins = json.load(f)
    
    # 过滤掉不属于任何类别的蛋白
    proteins = [protein for protein in proteins if any(value == 1 for value in protein["labels"].values())]
    
    # 初始化类别到蛋白质的映射（一个蛋白可能属于多个类别）
    category_to_proteins = defaultdict(list)
    for protein in proteins:
        for label, value in protein["labels"].items():
            if value == 1:
                category_to_proteins[label].append(protein)
    
    # 为每个蛋白初始化 split 字段（默认为空）
    for protein in proteins:
        protein["split"] = []
    
    # 按类别独立划分
    for label, protein_list in category_to_proteins.items():
        random.shuffle(protein_list)
        
        # 计算各集合的索引
        train_end = int(len(protein_list) * (1 - val_ratio - test_ratio))
        val_end = train_end + int(len(protein_list) * val_ratio)
        
        # 标记训练集、验证集和测试集
        for i, protein in enumerate(protein_list):
            if i < train_end:
                if "train" not in protein["split"]:
                    protein["split"].append("train")
            elif i < val_end:
                if "val" not in protein["split"]:
                    protein["split"].append("val")
            else:
                if "test" not in protein["split"]:
                    protein["split"].append("test")
    
    # 处理冲突：优先级 test > val > train
    for protein in proteins:
        if "test" in protein["split"]:
            protein["split"] = "test"
        elif "val" in protein["split"]:
            protein["split"] = "val"
        elif "train" in protein["split"]:
            protein["split"] = "train"
        else:
            protein["split"] = "train"  # 默认
    
    # 统计结果
    category_counts = defaultdict(lambda: {"train": 0, "val": 0, "test": 0})
    for protein in proteins:
        for label, value in protein["labels"].items():
            if value == 1:
                category_counts[label][protein["split"]] += 1
    
    # 打印详细的分布情况
    print("各类别样本分布：")
    total_counts = {"train": 0, "val": 0, "test": 0}
    for label, counts in sorted(category_counts.items()):
        print(f"{label}: train={counts['train']}, val={counts['val']}, test={counts['test']}")
        for split, count in counts.items():
            total_counts[split] += count
    
    print("\n总体样本分布：")
    print(f"训练集: {total_counts['train']} 样本")
    print(f"验证集: {total_counts['val']} 样本")
    print(f"测试集: {total_counts['test']} 样本")
    
    # 计算实际比例
    total_samples = sum(total_counts.values())
    print("\n实际划分比例：")
    print(f"训练集: {total_counts['train']/total_samples:.2%}")
    print(f"验证集: {total_counts['val']/total_samples:.2%}")
    print(f"测试集: {total_counts['test']/total_samples:.2%}")
    
    # 保存结果
    with open(outputjson, "w") as f:
        json.dump(proteins, f, indent=4)
    
    print(f"\n划分完成，结果保存至 {outputjson}")
    
    return proteins

def check_data_leakage(jsonfile):
    # 加载划分后的数据集
    with open(jsonfile, "r") as f:
        proteins = json.load(f)
    
    # 提取训练集和测试集中的蛋白质ID
    train_proteins = set()
    test_proteins = set()
    
    for protein in proteins:
        if protein["split"] == "train":
            train_proteins.add(protein["protein_id"])
        elif protein["split"] == "test":
            test_proteins.add(protein["protein_id"])
    
    # 检查是否有重复蛋白质
    leaked_proteins = train_proteins & test_proteins
    
    if leaked_proteins:
        print(f"数据泄露警告：{len(leaked_proteins)} 个蛋白质同时出现在训练集和测试集中。")
        print("泄露的蛋白质ID：", leaked_proteins)
    else:
        print("未检测到数据泄露。")


def validate_sequences(json_file):
    """检查JSON文件中所有序列是否包含非标准氨基酸字符"""
    with open(json_file) as f:
        data = json.load(f)
    
    invalid_chars = defaultdict(list)
    standard_aa = set("ACDEFGHIKLMNPQRSTVWYX")  # 标准氨基酸字符+未知字符X
    
    for i, item in enumerate(data):
        seq = item.get("sequence", "")
        bad_chars = set(seq) - standard_aa
        if bad_chars:
            for char in bad_chars:
                invalid_chars[char].append({
                    "index": i,
                    "protein_id": item.get("protein_id", ""),
                    "position": [pos+1 for pos, c in enumerate(seq) if c == char]
                })
    
    if invalid_chars:
        print("发现无效字符：")
        for char, records in invalid_chars.items():
            print(f"字符 '{char}' 出现在 {len(records)} 条序列中")
            print("示例记录：", records)  # 只显示前3条记录
        return False, invalid_chars
    else:
        print("所有序列均只包含标准氨基酸字符")
        return True, None




def split_train_val_data(json_file, val_ratio=0.2):
    """
    处理多标签情况的数据集划分
    """
    # 加载 JSON 文件
    with open(json_file, "r") as f:
        proteins = json.load(f)
    
    # 按所有活跃标签分组
    category_to_proteins = defaultdict(list)
    
    # 找到每个蛋白质的所有活跃标签
    for protein in proteins:
        # 找出值为1的所有标签
        active_labels = [
            label for label, value in protein["labels"].items() 
            if value == 1
        ]
        
        # 为每个活跃标签添加蛋白质
        for label in active_labels:
            category_to_proteins[label].append(protein)
    
    # 存储最终的训练集和验证集
    train_dataset = []
    val_dataset = []
    
    # 记录每个类别的样本数
    category_train_counts = defaultdict(int)
    category_val_counts = defaultdict(int)
    
    # 按类别划分训练集和验证集
    for label, protein_list in category_to_proteins.items():
        # 去重（同一个蛋白质可能属于多个类别）
        protein_list = list({p['protein_id']: p for p in protein_list}.values())
        
        # 随机打乱
        random.shuffle(protein_list)
        
        # 计算划分点
        total = len(protein_list)
        val_end = int(val_ratio * total)
        
        # 添加到训练集和验证集
        train_subset = protein_list[val_end:]
        val_subset = protein_list[:val_end]
        
        train_dataset.extend(train_subset)
        val_dataset.extend(val_subset)
        
        # 记录每个类别的样本数
        category_train_counts[label] = len(train_subset)
        category_val_counts[label] = len(val_subset)
    
    # 去重
    train_dataset = list({p['protein_id']: p for p in train_dataset}.values())
    val_dataset = list({p['protein_id']: p for p in val_dataset}.values())
    
    # 打印统计信息（类似上一个函数）
    print("\n数据集划分统计：")
    print(f"总样本数: {len(proteins)}")
    print(f"训练集样本数: {len(train_dataset)}")
    print(f"验证集样本数: {len(val_dataset)}")
    
    # 打印每个类别的样本数
    print("\n类别样本数统计：")
    print("类别 | 训练集 | 验证集 | 总数")
    print("-" * 30)
    
    # 按类别字母顺序排序输出
    for label in sorted(set(category_train_counts.keys()) | set(category_val_counts.keys())):
        train_count = category_train_counts.get(label, 0)
        val_count = category_val_counts.get(label, 0)
        total_count = train_count + val_count
        print(f"{label:20} | {train_count:6} | {val_count:6} | {total_count:6}")
    
    return train_dataset, val_dataset

def get_finetune_data(
    json_file, 
    split_label='val', 
    sampling_ratio=0.5,  # 从训练集抽取的比例
):
    """
    从JSON文件中提取数据并通过分层抽样平衡类别分布
    Args:
        json_file: JSON文件路径
        target_class: 目标类别，默认为 Nucleolus
        split_label: 数据集划分标签
        sampling_ratio: 从训练集抽取的比例
    Returns:
        平衡后的数据集和详细统计信息
    """
    # 加载 JSON 文件
    with open(json_file, "r") as f:
        proteins = json.load(f)
    
    # 按照标签分类数据集
    val_dataset = [
        protein for protein in proteins
        if protein.get('split', '') == split_label
    ]
    train_dataset = [
        protein for protein in proteins
        if protein.get('split', '') == 'train'
    ]
    
    # 统计原始训练集的类别分布
    train_class_distribution = {}
    for protein in train_dataset:
        for class_name, label in protein['labels'].items():
            if label == 1:
                train_class_distribution[class_name] = train_class_distribution.get(class_name, 0) + 1
    
    # 按类别分组训练数据
    train_class_groups = {}
    for protein in train_dataset:
        for class_name, label in protein['labels'].items():
            if label == 1:
                if class_name not in train_class_groups:
                    train_class_groups[class_name] = []
                train_class_groups[class_name].append(protein)
    
    # 打印原始分布
    print("\n原始训练集类别分布：")
    for cls, count in train_class_distribution.items():
        print(f"{cls}: {count} 个样本")
    
    # 分层抽样
    sampled_train_dataset = []
    for class_name, proteins in train_class_groups.items():
        # 计算当前类别需要抽取的样本数
        class_sample_count = int(len(proteins) * sampling_ratio)
        
        # 随机抽样
        class_samples = random.sample(proteins, class_sample_count)
        sampled_train_dataset.extend(class_samples)
    
    # 合并验证集和抽样的训练集
    final_dataset = val_dataset + sampled_train_dataset
    
    # 统计最终数据集的类别分布
    final_class_distribution = {}
    for protein in final_dataset:
        for class_name, label in protein['labels'].items():
            if label == 1:
                final_class_distribution[class_name] = final_class_distribution.get(class_name, 0) + 1
    
    # 打印最终分布
    print("\n最终数据集类别分布：")
    for cls, count in final_class_distribution.items():
        original_count = train_class_distribution.get(cls, 0)
        ratio = count / original_count if original_count > 0 else 0
        print(f"{cls}: {count} 个样本 (原始比例: {ratio:.2f})")
    
    # 打印数据集总体信息
    print(f"\n原始验证集样本数: {len(val_dataset)}")
    print(f"抽样的训练集样本数: {len(sampled_train_dataset)}")
    print(f"最终数据集总样本数: {len(final_dataset)}")
    
    return final_dataset
    
def get_finetune_test_data(json_file):
    """
    从JSON文件中提取数据并平衡类别分布
    Args:
        json_file: JSON文件路径
        target_class: 目标类别（默认为Nucleolus）
        balance_ratio: 正负样本允许的最大比例差异
        max_length: 最大序列长度阈值
    Returns:
        平衡后的数据集和详细统计信息
    """
    # 加载 JSON 文件
    with open(json_file, "r") as f:
        proteins = json.load(f)
    
    # 筛选标记为'val'的数据
    test_dataset = [
        protein for protein in proteins
        if protein.get('split', '') == 'test'
    ]
    print(f"\ntest_dataset 总样本数: {len(test_dataset)}")
    
    # 按类别分组和统计
    class_stats = {}
    for protein in test_dataset:
        for class_name, label in protein['labels'].items():
            if label == 1:
                if class_name not in class_stats:
                    class_stats[class_name] = []
                class_stats[class_name].append(protein)
    
    # 打印原始类别统计
    print("\n原始数据集类别统计：")
    for class_name, samples in class_stats.items():
        print(f"{class_name}: {len(samples)} 个样本")

    return test_dataset
def load_train_set(jsonfile):
    """
    从 JSON 文件中加载 split 为 "train" 的数据。

    参数:
        jsonfile (str): JSON 文件的路径。

    返回:
        list: 包含所有 split 为 "train" 的蛋白质数据的列表。
    """
    with open(jsonfile, "r") as f:
        data = json.load(f)
    
    # 过滤出 split 为 "train" 的数据
    train_set = [protein for protein in data if protein.get("split") == "train"]
    
    return train_set

def load_test_set(jsonfile):
    """
    从 JSON 文件中加载 split 为 "test" 的数据。

    参数:
        jsonfile (str): JSON 文件的路径。

    返回:
        list: 包含所有 split 为 "test" 的蛋白质数据的列表。
    """
    with open(jsonfile, "r") as f:
        data = json.load(f)
    
    # 过滤出 split 为 "test" 的数据
    test_set = [protein for protein in data if protein.get("split") == "test"]
    
    return test_set

def stratified_sampling(input_file, output_file, sample_size=1000):
    # 读取原始JSON数据
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # 筛选训练集数据
    train_data = [item for item in data if item['split'] == 'train']
    
    # 提取标签
    labels = [max(item['labels'], key=item['labels'].get) for item in train_data]
    
    # 获取唯一类别
    unique_classes = list(set(labels))
    
    # 计算每个类别的目标样本数
    class_counts = Counter(labels)
    total_samples = len(train_data)
    
    # 分层抽样
    stratified_indices, _ = train_test_split(
        range(total_samples), 
        train_size=sample_size, 
        stratify=labels, 
        random_state=42
    )
    
    # 抽样后的数据
    sampled_data = [train_data[i] for i in stratified_indices]
    
    # 统计抽样后的类别分布
    sampled_labels = [max(item['labels'], key=item['labels'].get) for item in sampled_data]
    sampled_class_counts = Counter(sampled_labels)
    
    # 打印类别分布
    print("原始训练集类别分布:")
    for cls in unique_classes:
        print(f"{cls}: {class_counts[cls]}")
    
    print("\n抽样后类别分布:")
    for cls in unique_classes:
        print(f"{cls}: {sampled_class_counts[cls]}")
    
    # 保存抽样数据
    with open(output_file, 'w') as f:
        json.dump(sampled_data, f, indent=4)
    
    # 返回抽样信息
    return {
        'original_counts': dict(class_counts),
        'sampled_counts': dict(sampled_class_counts)
    }

if __name__ == '__main__':


    # protgps 数据划分
    input_protpgs_file = "/mnt/data/fcc/binary_ESM/polyG/proteins_with_PolyG.json"
    output_protgps_file = "/mnt/data/fcc/binary_ESM/polyG/proteins_with_PolyG_split.json"

    # 数据集划分
    split_train_test_data_multi_label(input_protpgs_file,output_protgps_file)
    check_data_leakage(output_protgps_file)

    # 抽样遮蔽样本
    # output_mask_file = "/mnt/data/fcc/binary_ESM/dataset/uniprot&CDCODE&phasepdb_20250724_mask.json"
    # stratified_sampling(output_protgps_file, output_mask_file)



    # split_train_test_data_int("sae_18clusters.json")

    # validate_sequences(output_protgps_file)
