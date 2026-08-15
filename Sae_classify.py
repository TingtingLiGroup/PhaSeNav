import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from data_prepare import load_train_set,load_test_set
import esm
from dictionary import AutoEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, average_precision_score
import os
from datetime import datetime
from model_mlp import MLPModel, AttentionMLPModel
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from typing import Dict, Tuple  # 添加Tuple导入
from typing import Dict, List, Any
import pandas as pd
from train_classify import train_mlp_with_cv_multilabel, train_dynamic_feature_selection,evaluate_on_test_set_mlp,evaluate_all_1vs1_models_on_test,evaluate_on_test_set_mlp_from_pth,evaluate_all_1vs1_models_on_test_to_new_prots,evaluate_all_1vs1_models_on_test_from_pth
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
from collections import OrderedDict
import h5py
import gc


def get_residue_representations(train_set, model, batch_converter, alphabet, max_length=4096):
    """
    处理训练集并生成每段序列中每个残基的向量表示（自动GPU/CPU切换版）。
    
    参数:
        train_set (list): 训练集数据列表，每个元素为一个字典。
        model (torch.nn.Module): ESM 模型。
        batch_converter (function): 用于将序列转换为模型输入的批处理函数。
        max_length (int): 序列的最大长度，默认为 4096。
    
    返回:
        list: 每段序列的残基表示（已转移到CPU），每个元素为形状 [n_residues, feature_dim] 的张量。
        list: 样本ID列表。
        torch.Tensor: 形状为 (n, num_classes) 的标签矩阵（CPU）。
        list: 标签类别列表。
    """
    # 初始化设备设置
    gpu_device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    cpu_device = torch.device("cpu")
    current_device = gpu_device  # 默认使用GPU
    
    # 将模型移到默认设备
    model = model.to(current_device)
    
    all_residue_features = []  
    seq_ids = []
    labels = []
    label_categories = sorted(train_set[0]["labels"].keys()) if train_set else []
    num_classes = len(label_categories)
    print(f"开始处理数据集（默认设备: {current_device}）...")
    
    for protein in tqdm(train_set, total=len(train_set), desc="处理蛋白质序列"):
        seq_id = protein["protein_id"]
        seq = protein["sequence"]
        
        # 跳过超长序列
        seq_with_mask = seq.replace('<mask>', 'X')
        if len(seq_with_mask) > max_length:
            print(f"跳过样本 {seq_id}，序列长度 {len(seq_with_mask)} 超过 {max_length}")
            continue
            
        try:
            # 将序列转换为模型输入
            _, _, batch_tokens = batch_converter([(seq_id, seq)])
            batch_tokens = batch_tokens.to(current_device)
            
            batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
            
            with torch.no_grad():
                results = model(batch_tokens, repr_layers=[33], return_contacts=True)
            
            # 提取表示
            token_representations = results["representations"][33]
            
            # 处理每个序列
            for i, tokens_len in enumerate(batch_lens):
                residue_representations = token_representations[i, 1 : tokens_len - 1]
                all_residue_features.append(residue_representations.cpu())
                seq_ids.append(seq_id)
                protein_labels = [protein["labels"][category] for category in label_categories]
                labels.append(protein_labels)
                
            # 如果之前切换到CPU，现在切换回GPU
            if current_device == cpu_device:
                current_device = gpu_device
                model = model.to(current_device)
                print(f"切换回GPU设备: {current_device}")
                
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print(f"GPU内存不足，尝试切换到CPU处理序列 {seq_id} (长度: {len(seq_with_mask)})")
                current_device = cpu_device
                model = model.to(current_device)
                
                # 释放GPU内存
                if 'batch_tokens' in locals():
                    del batch_tokens
                if 'results' in locals():
                    del results
                torch.cuda.empty_cache()
                
                # 重试处理当前序列
                try:
                    _, _, batch_tokens = batch_converter([(seq_id, seq)])
                    batch_tokens = batch_tokens.to(current_device)
                    
                    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
                    
                    with torch.no_grad():
                        results = model(batch_tokens, repr_layers=[33], return_contacts=True)
                    
                    token_representations = results["representations"][33]
                    
                    for i, tokens_len in enumerate(batch_lens):
                        residue_representations = token_representations[i, 1 : tokens_len - 1]
                        all_residue_features.append(residue_representations.cpu())
                        seq_ids.append(seq_id)
                        protein_labels = [protein["labels"][category] for category in label_categories]
                        labels.append(protein_labels)
                        
                except Exception as e:
                    print(f"CPU处理 {seq_id} 时出错: {e}")
            else:
                print(f"处理 {seq_id} 时出错: {e}")
        finally:
            # 清理当前序列的内存
            if 'batch_tokens' in locals():
                del batch_tokens
            if 'results' in locals():
                del results
            if 'token_representations' in locals():
                del token_representations
            torch.cuda.empty_cache()
    
    # 将标签列表转换为张量
    label_matrix = torch.tensor(labels, dtype=torch.float32)
    return all_residue_features, seq_ids, label_matrix, label_categories

def save_tensors_residue(residue_representations, seq_ids, label_matrix, save_dir="./saved_tensors", prefix="train", include_timestamp=True):
    """
    保存残基级表示、标签矩阵和序列ID到指定目录。

    参数:
        residue_representations (list): 每段序列的残基表示，每个元素为形状 [n_residues, feature_dim] 的张量。
        seq_ids (list): 样本ID列表。
        save_dir (str): 存储目录路径，默认当前目录下的 saved_tensors。
        prefix (str): 文件名前缀，默认为 "train"。
        include_timestamp (bool): 是否在文件名中包含时间戳，默认 True。
    """
    
    # 自动创建目录
    os.makedirs(save_dir, exist_ok=True)

    # 生成带时间戳的文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") if include_timestamp else ""
    base_name = f"{prefix}_tensors_{timestamp}" if timestamp else f"{prefix}_tensors"

    # 保存为 PyTorch 文件
    save_path = os.path.join(save_dir, f"{base_name}.pt")
    torch.save({
        'residue_representations': residue_representations,  # 残基级表示
        'seq_ids': seq_ids,  # 序列ID
        'label_matrix': label_matrix
    }, save_path)

    # 打印保存信息
    print(f"张量已保存至: {os.path.abspath(save_path)}")
    print(f"文件大小: residue_representations {sum(r.element_size() * r.nelement() for r in residue_representations) / 1024**2:.2f} MB")

def load_tensors(path, device="cuda:1"):
    """
    加载保存的 train_plm_rep 和 label_matrix

    参数:
        path (str): 保存的 .pt 文件路径
        device (str): 加载后张量的设备，默认 "cuda"

    返回:
        train_plm_rep (torch.Tensor): 形状为 (n, 1280) 的特征矩阵
        label_matrix (torch.Tensor): 形状为 (n, 18) 的标签矩阵
    """
    data = torch.load(path, map_location=torch.device(device))
    return data['train_plm_rep'], data['label_matrix']

def load_tensors_residue(path, device="cuda"):
    """
    加载保存的 train_plm_rep 和 seq_ids

    参数:
        path (str): 保存的 .pt 文件路径
        device (str): 加载后张量的设备，默认 "cuda"

    返回:
        train_plm_rep (torch.Tensor): 形状为 (n, 1280) 的特征矩阵
    """
    data = torch.load(path, map_location=torch.device(device))
    return data['residue_representations'], data['seq_ids']

def multi_label_analysis(label_matrix):
    """
    多标签数据分析
    
    Args:
        label_matrix (torch.Tensor): 标签矩阵
    """
    print("多标签数据分析:")
    num_classes = label_matrix.shape[1]
    
    # 每个类别的样本分布
    class_distribution = []
    for class_idx in range(num_classes):
        pos_samples = label_matrix[:, class_idx].sum().item()
        total_samples = len(label_matrix)
        pos_ratio = pos_samples / total_samples
        
        class_distribution.append({
            'class': class_idx,
            'positive_samples': pos_samples,
            'total_samples': total_samples,
            'positive_ratio': pos_ratio
        })
    
    # 按正样本数排序
    class_distribution.sort(key=lambda x: x['positive_samples'])
    
    print("\n类别样本分布:")
    for cls in class_distribution:
        print(f"类别 {cls['class']}: "
              f"总样本 {cls['total_samples']}, "
              f"正样本 {cls['positive_samples']} "
              f"({cls['positive_ratio']:.2%})")
    
    # 标签共现分析
    label_co_occurrence = label_matrix.T @ label_matrix
    print("\n标签共现矩阵:")
    print(label_co_occurrence)
    
    return class_distribution



def process_sequence_representations(representations, sae_model, batch_size=64):
    """
    处理序列表示并计算激活值
    
    参数:
        representations: 形状为 [n, 1280] 的张量，每一行表示一条序列的向量表示
        sae_model: 加载的SAE模型
        batch_size: 批处理大小
        
    返回:
        torch.Tensor: 形状为 [n, 40960] 的稀疏特征矩阵
    """
    sae_model.eval()
    device = next(sae_model.parameters()).device  # 获取模型所在的设备
    all_activations = []
    
    with torch.no_grad():
        # 将输入数据移动到设备
        representations = representations.to(device)
        
        # 分批处理以避免内存不足
        for i in range(0, len(representations), batch_size):
            batch = representations[i:i + batch_size]  # [batch_size, 1280]
            batch_activations = sae_model.encode(batch)  # [batch_size, 40960]
            all_activations.append(batch_activations.cpu())  # 移回CPU
        
        # 合并所有批次结果
        activations = torch.cat(all_activations, dim=0)  # [n, 40960]
    
    return activations

def process_sequence_representations_residue(protein_list, sae_model, batch_size=64):
    """
    处理蛋白质序列列表，对每个氨基酸进行SAE编码并池化
    
    参数:
        protein_list: 长度为n的列表，每个元素为[L, 1280]的氨基酸特征矩阵
        sae_model: 加载的SAE模型
        batch_size: 批处理大小（针对氨基酸数量）
        
    返回:
        torch.Tensor: 形状为[n, 40960]的稀疏特征矩阵（每个蛋白质的池化后表示）
    """
    sae_model.eval()
    device = next(sae_model.parameters()).device
    all_sequence_reps = []
    
    with torch.no_grad():
        for protein in protein_list:
            # 当前蛋白质的氨基酸特征 [L, 1280]
            amino_features = protein.to(device)
            L = amino_features.shape[0]
            
            # 分批处理氨基酸以避免内存不足
            protein_activations = []
            for i in range(0, L, batch_size):
                batch = amino_features[i:i + batch_size]  # [current_batch_size, 1280]
                batch_activations = sae_model.encode(batch)  # [current_batch_size, 40960]
                protein_activations.append(batch_activations)
            
            # 合并当前蛋白质的所有氨基酸编码 [L, 40960]
            protein_encoded = torch.cat(protein_activations, dim=0)
            
            # 平均池化得到序列级表示 [40960,]
            sequence_rep = protein_encoded.mean(dim=0)
            all_sequence_reps.append(sequence_rep.cpu())  # 移回CPU暂存
        
        # 合并所有蛋白质的序列表示 [n, 40960]
        sparse_features = torch.stack(all_sequence_reps)
    
    return sparse_features

def process_sequence_representations_residue_nonpool(protein_list, sae_model, batch_size=64):
    """
    处理蛋白质序列列表，对每个氨基酸进行SAE编码
    
    参数:
        protein_list: 长度为n的列表，每个元素为[L, 1280]的氨基酸特征矩阵
        sae_model: 加载的SAE模型
        batch_size: 批处理大小（针对氨基酸数量）
        
    返回:
        list: 长度为n的列表，每个元素为[L, 40960]的氨基酸特征矩阵
    """
    sae_model.eval()
    device = next(sae_model.parameters()).device
    all_sequence_reps = []
    
    with torch.no_grad():
        for protein in protein_list:
            # 当前蛋白质的氨基酸特征 [L, 1280]
            amino_features = protein.to(device)
            L = amino_features.shape[0]
            
            # 分批处理氨基酸以避免内存不足
            protein_activations = []
            for i in range(0, L, batch_size):
                batch = amino_features[i:i + batch_size]  # [current_batch_size, 1280]
                batch_activations = sae_model.encode(batch)  # [current_batch_size, 40960]
                protein_activations.append(batch_activations)
            
            # 合并当前蛋白质的所有氨基酸编码 [L, 40960]
            protein_encoded = torch.cat(protein_activations, dim=0).cpu()
            all_sequence_reps.append(protein_encoded)
    
    return all_sequence_reps


def process_and_save_sae_features(
    protein_list, sae_model, output_dir, batch_size=64, save_every=1000, start_index=0
):
    """
    对蛋白质列表进行SAE编码，并每1000个蛋白保存为一个pt文件。

    参数:
        protein_list: List[Tensor]，每个元素是[L, 1280]的氨基酸特征
        sae_model: SAE模型
        output_dir: 保存编码结果的文件夹
        batch_size: 氨基酸级别批大小
        save_every: 每多少个蛋白保存一个pt文件
        start_index: 文件命名起始编号（适合断点续跑）
    """
    os.makedirs(output_dir, exist_ok=True)
    sae_model.eval()
    device = next(sae_model.parameters()).device

    current_batch = {}
    file_count = start_index
    pbar = tqdm(enumerate(protein_list), total=len(protein_list), desc="Encoding proteins")

    with torch.no_grad():
        for idx, protein in pbar:
            # 将蛋白质特征移动到 GPU
            amino_features = protein.to(device)
            L = amino_features.shape[0]

            # 分批处理 [L, 1280] → [L, 40960]
            protein_activations = []
            for i in range(0, L, batch_size):
                batch = amino_features[i:i + batch_size]
                batch_activations = sae_model.encode(batch)
                protein_activations.append(batch_activations)

            # 合并成一个蛋白质的 [L, 40960]
            protein_encoded = torch.cat(protein_activations, dim=0).cpu()

            # 存入当前batch（以索引命名）
            current_batch[f"protein_{idx}"] = protein_encoded

            # 每达到save_every保存一次
            if (idx + 1) % save_every == 0:
                save_path = os.path.join(output_dir, f"batch_{file_count}.pt")
                torch.save(current_batch, save_path)
                current_batch = {}
                file_count += 1
                torch.cuda.empty_cache()

        # 保存最后一批不足1000的蛋白
        if current_batch:
            save_path = os.path.join(output_dir, f"batch_{file_count}.pt")
            torch.save(current_batch, save_path)
            torch.cuda.empty_cache()

    print(f"\n编码完成，总共保存 {file_count + 1 - start_index} 个文件。")


def save_best_feature_selection_results(
    best_results: Dict,
    output_dir: str = './best_feature_selection_results'
):
    """
    保存动态特征选择的最佳结果，包括模型性能、特征重要性等
    Args:
        best_results (Dict): 最佳迭代结果
        output_dir (str): 结果保存目录
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化保存路径变量
    performance_path = None
    feature_importance_path = None
    summary_path = None
    selected_features_path = None
    
    # 1. 保存模型性能细节
    if 'best_results' in best_results:
        # 处理二分类结果性能数据
        performance_data = []
        # for result in best_results['best_results']['binary_results']:
        #     performance_data.append({
        #         'negative_class': result.get('negative_class', -1),
        #         'roc_auc': result.get('roc_auc', 0),
        #         'accuracy': result.get('accuracy', 0),
        #         'precision': result.get('precision', 0),
        #         'recall': result.get('recall', 0),
        #         'f1_score': result.get('f1_score', 0),
        #         'model_path': result.get('model_path', '')
        #     })
        
        test_results = best_results.get("test_results", {})

        # 直接从 test_results_1vs1 构造 performance_data
        performance_data = []
        for (cls_a, cls_b), metrics in test_results.items():
            performance_data.append({
                'class_pair': f'{cls_a}_vs_{cls_b}',
                'roc_auc': metrics.get('AUC', 0),
                'accuracy': metrics.get('accuracy', 0),
                'precision': metrics.get('precision', 0),
                'recall': metrics.get('recall', 0),
                'f1_score': metrics.get('f1_score', 0)
            })


        # 创建性能DataFrame
        performance_df = pd.DataFrame(performance_data)
        performance_path = os.path.join(output_dir, 'best_model_performance.csv')
        performance_df.to_csv(performance_path, index=False)
        
        # 获取平均AUC
        # mean_auc = best_results.get('best_performance', 0)
        # 使用测试集 AUC 均值
        auc_values = [v['AUC'] for v in test_results.values()]
        mean_auc = float(np.mean(auc_values)) if auc_values else 0
        
        # 处理特征重要性
        feature_importance = best_results.get('best_feature_importances')
        best_feature_masks = best_results.get('best_feature_masks')
        
        if feature_importance is not None and best_feature_masks is not None:
            # 获取被选择的特征的索引和重要性
            selected_feature_indices = torch.nonzero(best_feature_masks).squeeze()
            selected_feature_importances = feature_importance[selected_feature_indices]
            
            # 按重要性绝对值降序排序
            importance_abs = torch.abs(selected_feature_importances)
            sorted_indices = torch.argsort(importance_abs, descending=True)
            
            selected_features_df = pd.DataFrame({
                'feature_index': selected_feature_indices[sorted_indices].cpu().numpy(),
                'importance_value': selected_feature_importances[sorted_indices].cpu().numpy(),
                'abs_importance': importance_abs[sorted_indices].cpu().numpy()
            })
            
            selected_features_path = os.path.join(output_dir, 'selected_features.csv')
            selected_features_df.to_csv(selected_features_path, index=False)
            
            # 特征重要性 Top 1000
            top_values, top_indices = torch.topk(torch.abs(feature_importance), 1000)
            feature_importance_df = pd.DataFrame({
                'feature_index': top_indices.cpu().numpy(),
                'importance_value': top_values.cpu().numpy(),
                'signed_importance': feature_importance[top_indices].cpu().numpy()
            })
            feature_importance_path = os.path.join(output_dir, 'top_features_importance.csv')
            feature_importance_df.to_csv(feature_importance_path, index=False)
        
        # 特征选择摘要
        feature_selection_summary = pd.DataFrame({
            'metric': [
                'Mean AUC',
                'Total Features',
                'Selected Features',
                'Selection Ratio'
            ],
            'value': [
                mean_auc,
                feature_importance.shape[0] if feature_importance is not None else 0,
                len(selected_feature_indices) if 'selected_feature_indices' in locals() else 0,
                (len(selected_feature_indices) / feature_importance.shape[0]) 
                    if feature_importance is not None and 'selected_feature_indices' in locals()
                    else 0
            ]
        })
        summary_path = os.path.join(output_dir, 'feature_selection_summary.csv')
        feature_selection_summary.to_csv(summary_path, index=False)
    
    return {
        'performance_path': performance_path,
        'feature_importance_path': feature_importance_path,
        'summary_path': summary_path,
        'selected_features_path': selected_features_path,
        'output_dir': output_dir
    }

def save_final_feature_selection_with_test_auc(
    best_results: Dict,
    test_auc: float,
    output_dir: str = './best_feature_selection_results',
    class_idx: int = 0
) -> Dict:
    """
    保存最终动态特征选择的结果，包括测试集性能（AUC）、特征重要性等。
    
    Args:
        best_results (Dict): 包含掩码、重要性等的结果字典。
        test_auc (float): 在测试集上计算出的 AUC。
        output_dir (str): 保存目录。
    """
    os.makedirs(output_dir, exist_ok=True)

    # 初始化保存路径变量
    feature_importance_path = None
    summary_path = None
    selected_features_path = None

    # 获取特征重要性
    feature_importance = best_results.get('best_feature_importances')
    best_feature_masks = best_results.get('best_feature_masks')

    if feature_importance is not None and best_feature_masks is not None:
        # 被选中特征索引和重要性
        selected_feature_indices = torch.nonzero(best_feature_masks).squeeze()
        selected_feature_importances = feature_importance[selected_feature_indices]

        # 按绝对值降序排序
        importance_abs = torch.abs(selected_feature_importances)
        sorted_indices = torch.argsort(importance_abs, descending=True)

        selected_features_df = pd.DataFrame({
            'feature_index': selected_feature_indices[sorted_indices].cpu().numpy(),
            'importance_value': selected_feature_importances[sorted_indices].cpu().numpy(),
            'abs_importance': importance_abs[sorted_indices].cpu().numpy()
        })

        selected_features_path = os.path.join(output_dir, f'selected_features_{class_idx}.csv')
        selected_features_df.to_csv(selected_features_path, index=False)

        # 特征重要性 Top 1000（完整范围内）
        top_values, top_indices = torch.topk(torch.abs(feature_importance), 1000)
        feature_importance_df = pd.DataFrame({
            'feature_index': top_indices.cpu().numpy(),
            'importance_value': top_values.cpu().numpy(),
            'signed_importance': feature_importance[top_indices].cpu().numpy()
        })

        feature_importance_path = os.path.join(output_dir, f'top_features_importance_{class_idx}.csv')
        feature_importance_df.to_csv(feature_importance_path, index=False)

        # 摘要
        feature_selection_summary = pd.DataFrame({
            'metric': [
                'Test AUC',
                'Total Features',
                'Selected Features',
                'Selection Ratio'
            ],
            'value': [
                test_auc,
                feature_importance.shape[0],
                len(selected_feature_indices),
                len(selected_feature_indices) / feature_importance.shape[0]
            ]
        })

        summary_path = os.path.join(output_dir, f'feature_selection_summary_{class_idx}.csv')
        feature_selection_summary.to_csv(summary_path, index=False)

    return {
        'test_auc': test_auc,
        'feature_importance_path': feature_importance_path,
        'summary_path': summary_path,
        'selected_features_path': selected_features_path,
        'output_dir': output_dir
    }

def save_batch_info(save_dir, prefix, processed_batches):
    """保存已处理的批次信息"""
    with open(os.path.join(save_dir, f"{prefix}_processed_batches.json"), 'w') as f:
        json.dump(processed_batches, f)

def load_batch_info(save_dir, prefix):
    """加载已处理的批次信息"""
    try:
        with open(os.path.join(save_dir, f"{prefix}_processed_batches.json"), 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return []

def load_saved_batches(save_dir, prefix):
    """加载已保存的批次结果"""
    all_plm_reps = []
    all_seq_ids = []
    all_labels = []
    
    # 遍历保存目录，加载所有批次文件
    for filename in os.listdir(save_dir):
        if filename.startswith(f"{prefix}_batch_") and filename.endswith(".pt"):
            batch_data = torch.load(os.path.join(save_dir, filename))
            all_plm_reps.extend(batch_data["plm_reps"])
            all_seq_ids.extend(batch_data["seq_ids"])
            all_labels.append(batch_data["label_matrix"])
    
    # 合并标签矩阵
    if all_labels:
        full_label_matrix = torch.cat(all_labels, dim=0)
    else:
        full_label_matrix = None
    
    return all_plm_reps, all_seq_ids, full_label_matrix

def process_in_batches(sequences, model, batch_converter, alphabet, save_dir, prefix, batch_size=50, max_length=4096):
    """
    分批处理序列并保存结果（支持断点续传，合并所有批次结果）
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 加载已处理的批次信息
    processed_batches = load_batch_info(save_dir, prefix)
    start_idx = len(processed_batches) * batch_size  # 计算起始位置
    
    # 加载已保存的批次结果
    all_plm_reps, all_seq_ids, full_label_matrix = load_saved_batches(save_dir, prefix)
    
    # 分批处理
    for i in range(start_idx, len(sequences), batch_size):
        batch = sequences[i:i + batch_size]
        batch_num = i // batch_size
        print(f"\n处理批次 {batch_num + 1}/{(len(sequences)-1)//batch_size + 1} (序列 {i} 到 {min(i+batch_size, len(sequences))-1})")
        
        # 处理当前批次
        plm_rep, seq_ids, label_matrix, _ = get_residue_representations(
            batch, model, batch_converter, alphabet, max_length=max_length
        )
        
        # 保存当前批次
        batch_prefix = f"{prefix}_batch_{batch_num}"
        batch_data = {
            "plm_reps": plm_rep,
            "seq_ids": seq_ids,
            "label_matrix": label_matrix
        }
        torch.save(batch_data, os.path.join(save_dir, f"{batch_prefix}.pt"))
        
        # 记录已处理的批次
        processed_batches.append(batch_num)
        save_batch_info(save_dir, prefix, processed_batches)
        
        # 收集当前批次结果
        all_plm_reps.extend(plm_rep)
        all_seq_ids.extend(seq_ids)
        if full_label_matrix is None:
            full_label_matrix = label_matrix
        else:
            full_label_matrix = torch.cat([full_label_matrix, label_matrix], dim=0)
    
    # 保存完整结果
    if full_label_matrix is not None:
        save_tensors_residue(
            residue_representations=all_plm_reps,
            seq_ids=all_seq_ids,
            label_matrix=full_label_matrix,
            save_dir=save_dir,
            prefix=prefix,
            include_timestamp=False
        )
    
    return all_plm_reps, all_seq_ids, full_label_matrix

def save_label_categories(train_set, filepath="label_categories.json"):
    """
    将标签类别保存为带有数字索引的JSON文件
    
    参数:
        label_categories (list): 标签类别列表
        filepath (str): 保存路径，默认为"label_categories.json"
    
    返回:
        dict: 生成的类别到数字的映射字典
    """

    label_categories = sorted(train_set[0]["labels"].keys()) if train_set else []

    # 创建有序的类别到数字的映射 (从0开始)
    category_to_index = OrderedDict(
        (category, idx) for idx, category in enumerate(label_categories)
    )
    
    # 保存到JSON文件
    with open(filepath, 'w') as f:
        json.dump(category_to_index, f, indent=4)
    
    print(f"标签类别映射已保存到: {filepath}")
    return category_to_index

def save_scores_to_csv(protein_scores: List[Tuple[str, float]], output_file: str):
    df = pd.DataFrame(protein_scores, columns=['Protein_ID', 'Score'])
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")
if __name__ == '__main__':

# ####################################################################################
# 1、这部分开始处理原始数据文件，通过esm获取序列的表示，并且保存pt格式到save_dir
    # print("加载 PLM 模型...")
    # model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    # batch_converter = alphabet.get_batch_converter()
    # model.eval()
    # print("PLM 模型加载完成\n")
    # json_file = '/mnt/data/fcc/binary_ESM/negtive_seg_select/mut_appended_sequences.json'

    # sequences_train = load_train_set(json_file)
    # sequences_test = load_test_set(json_file)

    # category_mapping = save_label_categories(sequences_train,filepath="label_categories_uniprot&CDCODE&phasepdb_20250724.json")

    
    # 处理训练集
    print("\n开始处理训练集...")
    # plm_rep_train, seq_ids_train, label_matrix_train = process_in_batches(
    #     sequences_train, model, batch_converter,alphabet, 
    #     save_dir="./polyG/batch_full_results_20260521/my_saved_tensors/3072", 
    #     prefix="train",
    #     batch_size=100,  # 可调整
    #     max_length=3072
    # )

    print("\n开始处理测试集...")
    # plm_rep_test, seq_ids_test, label_matrix_test = process_in_batches(
    #     sequences_test, model, batch_converter,alphabet,
    #     save_dir="./negtive_seg_select/my_saved_tensors/3072", 
    #     prefix="test",
    #     batch_size=100,  # 可调整
    #     max_length=2900
    # )
    
    # print("plm_rep_train 形状:", len(plm_rep_train))
    # print("label_matrix_train 形状:", len(label_matrix_train))
    # print("seq_ids_train 长度:", len(seq_ids_train))
    # print("plm_rep_test 形状:", len(plm_rep_test))
    # print("label_matrix_test 形状:", len(label_matrix_test))
    # print("seq_ids_test 长度:", len(seq_ids_test))
    
# ####################################################################################
# 2、这部分自动从第一步保存的save_dir中读取序列表示，然后使用sae对序列稀疏化，保存pt文件到train_pt_file
    # 分类模型训练
    
    # train_data = torch.load("./polyG/batch_full_results_20260521/my_saved_tensors/3072/train_tensors.pt")
    # train_plm_rep = train_data['residue_representations']
    # train_seq_ids = train_data['seq_ids']
    # train_label_matrix = train_data['label_matrix']

    test_data = torch.load("./my_saved_tensors/uniprot&CDCODE&phasepdb_20250724/3072/test_tensors.pt")
    test_plm_rep = test_data['residue_representations']
    test_seq_ids = test_data['seq_ids']
    test_label_matrix = test_data['label_matrix']

    # # print(len(train_plm_rep))  # 应为 [100, 40960]
    # # print(train_label_matrix.shape)

    # print(len(test_plm_rep))  # 应为 [100, 40960]
    # print(test_label_matrix.shape)

    # 检查并设置设备
    # device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    # print(f"当前设备: {device}")

    # # 1. 加载SAE模型
    # print("加载 SAE 模型...")
    # chk_path = "step_80000.pt"
    # sae = AutoEncoder.from_pretrained(chk_path)
    # sae.eval()  # 切换到评估模式（禁用dropout等）

    # 2. 设备设置
    # sae = sae.to(device)

    # # 处理序列表示
    # print("开始处理蛋白质序列...")
    # train_pt_file = "./polyG/batch_full_results_20260521/protein_reps_polyg_pt/train"
    test_pt_file = "./protein_reps_uniprot&CDCODE&phasepdb_20250724_pt/test"

#     process_and_save_sae_features(
#     protein_list=train_plm_rep,
#     sae_model=sae,
#     output_dir=train_pt_file,
#     batch_size=64,
#     save_every=500
# )
    
#     process_and_save_sae_features(
#     protein_list=test_plm_rep,
#     sae_model=sae,
#     output_dir=test_pt_file,
#     batch_size=64,
#     save_every=500
# )
# ####################################################################################

# 3、这部分是训练过程，根据步骤一获得的label_categories_uniprot&CDCODE&phasepdb_20250826.json，选择一个名称给group_name，针对这个group进行1vrest训练
    # 训练过程
    # 1v1 train
    # print("训练测试数据加载完成\n")

    # group_name = "PolyG"
    # mapping_file = "label_categories_proteins_with_PolyG_split.json"
    # with open(mapping_file) as f:
    #     label_map = json.load(f)
    #     class_idx = label_map[group_name]  # 直接通过键名获取对应的数字值
    # print(class_idx)

    # # 设置参数范围
    # lr = 0.0001
    # base_lambda_l1_values = [0.000001]  # L1惩罚项的范围

    # for base_lambda_l1_value in  base_lambda_l1_values:
    #     iteration_results = train_dynamic_feature_selection(
    #         pt_folder=train_pt_file,
    #         label_matrix=train_label_matrix,
    #         max_iterations=10,
    #         final_feature_num=3000,
    #         class_idx=class_idx,
    #         lr=lr,                  # 将学习率传递给函数
    #         base_lambda_l1=base_lambda_l1_value  # 将L1惩罚项传递给函数
    #     )

    # # 如果用GPU，释放显存缓存
    # if torch.cuda.is_available():
    #     torch.cuda.empty_cache()

# ####################################################################################
    # 测试过程

#     test_results_1vs1 = evaluate_all_1vs1_models_on_test(
#     pt_folder=test_pt_file,
#     label_matrix_test=test_label_matrix,
#     best_feature_mask=iteration_results["best_feature_masks"],
#     binary_results=iteration_results["best_results"]["binary_results"],
#     input_dim=40960,
#     device="cuda:1"
# )

#     for (cls_a, cls_b), metrics in test_results_1vs1.items():
#         print(f"{cls_a} vs {cls_b} → AUC: {metrics['AUC']:.4f}")
#     # 保存结果
#     # 1vs1
#     saved_results = save_best_feature_selection_results(
#         best_results={
#             'best_iteration': iteration_results['best_iteration'],
#             'best_performance': iteration_results['best_performance'],
#             'best_feature_masks': iteration_results['best_feature_masks'],
#             'best_results': iteration_results['best_results'],
#             'best_feature_importances': iteration_results['best_feature_importances'],
#             'test_results': test_results_1vs1  # ✅ 加这一项
#         },
#         output_dir=f"./final_test_results_class_20260522_{group_name}"
#     )

    # 1vsrest
    # test_results = evaluate_on_test_set_mlp(
    # pt_folder=test_pt_file,
    # label_matrix_test=test_label_matrix,
    # best_feature_mask=iteration_results["best_feature_masks"],
    # model_state_dict=iteration_results["best_results"]["model_state"],
    # input_dim=40960,  # 根据你原始特征维度
    # class_idx= class_idx
    # )
    # print(test_results["AUC"])
    # print("✅ Using best iteration:", iteration_results["best_iteration"])
    # # # 1vsrest
    # save_final_feature_selection_with_test_auc(
    #     best_results=iteration_results,
    #     test_auc=test_results["AUC"],
    #     output_dir=f"./final_test_results_class_Uniport+CD-CODE+phasepdb_{group_name}",
    #     class_idx = class_idx
    # )


    # ####################################################################################
    # 文件读取测试
    # 1vs1
    # 读取 CSV 文件
    for i in range(13,14): 
        class_id = i
        csv_file_path = f'./final_results_20250724/selected_features_{class_id}.csv'  # 替换为您的 CSV 文件路径
        df = pd.read_csv(csv_file_path)

        # 提取特征 ID，并将其转换为整数
        feature_indices = df['feature_index'].astype(int).tolist()

        # 创建一个长度为 40960 的零掩码
        feature_dim = 40960
        binary_mask = torch.zeros(feature_dim, dtype=torch.int)  # 使用 int 类型创建张量

        # 将特征 ID 对应的位置设置为 1
        for index in feature_indices:
            if index < feature_dim:  # 确保索引在有效范围内
                binary_mask[index] = 1

        # 打印结果
        print("Binary Mask:", binary_mask)
        results = evaluate_all_1vs1_models_on_test_from_pth(
            pt_folder=test_pt_file,
            label_matrix_test=test_label_matrix,
            best_feature_mask=binary_mask,
            model_folder=f"./dynamic_feature_models/iteration_8/class_{class_id}",
            input_dim=40960,
            device="cuda:1"
        )

        # 打印单个模型和平均结果
        if results:
            # 打印单个模型结果
            for (cls_a, cls_b), metrics in results['evaluation_summary'].items():
                print(
                    f"{cls_a} vs {cls_b} → "
                    f"AUC: {metrics['AUC']:.4f} | "
                    f"AP: {metrics['AP']:.4f} | "
                    f"Acc: {metrics['Accuracy']:.4f} | "
                    f"F1: {metrics['F1']:.4f}"
                )
            
            # 打印平均结果
            print(f"\n📊 总体平均结果:")
            print(f"  AUC: {results['avg_auc']:.4f} ± {results['std_auc']:.4f}")
            print(f"  AP:  {results['avg_ap']:.4f} ± {results['std_ap']:.4f}")
            print(f"  F1:  {results['avg_f1']:.4f} ± {results['std_f1']:.4f}")

        # 保存结果
        rows = []
        for (class_a, class_b), metrics in results["evaluation_summary"].items():
            rows.append({
                "class_pair": f"{class_a}_vs_{class_b}",
                "roc_auc":    metrics["AUC"],
                "ap":         metrics["AP"],        # ← 新增
                "accuracy":   metrics["Accuracy"],
                "f1_score":   metrics["F1"],
            })

        # 追加汇总行
        rows.append({
            "class_pair": "average",
            "roc_auc":    results["avg_auc"],
            "ap":         results["avg_ap"],        # ← 新增
            "accuracy":   np.mean([m["Accuracy"] for m in results["evaluation_summary"].values()]),
            "f1_score":   results["avg_f1"],
        })

        df = pd.DataFrame(rows)
        df.to_csv(f"best_model_performance_{class_id}.csv", index=False)
    # 1vs1

    # 1vrest测试

    
#     model_filepath = './dynamic_feature_models/iteration_1/class_6/class_6_vs_rest_best_model.pth'

#     evaluation_results = evaluate_on_test_set_mlp_from_pth(
#         model_filepath = model_filepath,
#     pt_folder=test_pt_file,
#     label_matrix_test=test_label_matrix,
#     best_feature_mask=binary_mask,
#     input_dim=40960,
#     batch_size=64,
#     class_idx = class_idx,
#     device="cuda:1"
# )
#     print(evaluation_results["AUC"])

    # ####################################################################################
    # 对review蛋白进行预测

# #     # 读取 CSV 文件
#     csv_file_path = './final_results_20250724/selected_features_4.csv'  # 替换为您的 CSV 文件路径
#     df = pd.read_csv(csv_file_path)

#     # 提取特征 ID，并将其转换为整数
#     feature_indices = df['feature_index'].astype(int).tolist()

#     # 创建一个长度为 40960 的零掩码
#     feature_dim = 40960
#     binary_mask = torch.zeros(feature_dim, dtype=torch.int)  # 使用 int 类型创建张量

#     # 将特征 ID 对应的位置设置为 1
#     for index in feature_indices:
#         if index < feature_dim:  # 确保索引在有效范围内
#             binary_mask[index] = 1

#     # review蛋白筛选
#     model_filepath = './dynamic_feature_models/iteration_4/class_4'
#     # 需要被review的蛋白准备
#     test_pt_file = "./negtive_seg_select/protein_reps_mut_pt/test"
#     train_auc_file_path = "./final_results_20250724/best_model_performance_4.csv"

#     # 读取 CSV 文件
#     auc_data = pd.read_csv(train_auc_file_path)

#     # 提取 class_pair 和 roc_auc 列
#     auc_results = auc_data[['class_pair', 'roc_auc']]
    

#     # 如果需要将结果按字典的形式存储
#     train_auc = auc_results.set_index('class_pair').to_dict()['roc_auc']
#     evaluation_results = evaluate_all_1vs1_models_on_test_to_new_prots(
#         model_filepath = model_filepath,
#     pt_folder=test_pt_file,
#     best_feature_mask=binary_mask,
#     train_auc=train_auc,
#     input_dim=40960,
#     batch_size=64,
#     device="cuda:1"
# )

#     # 保存结果
#     save_scores_to_csv(evaluation_results, 'final_protein_scores.csv')  # 保存为 CSV 文件
    

#     test_data = torch.load("./negtive_seg_select/my_saved_tensors/3072/test_tensors.pt")
#     test_seq_ids = test_data['seq_ids']
#     print(len(test_seq_ids))

#     # 读取现有的 protein_scores CSV 文件
#     input_csv_path = 'final_protein_scores.csv'
#     df = pd.read_csv(input_csv_path)

#     # 确保 test_seq_ids 列表和 Protein_ID 列之间的长度匹配
#     if len(df) != len(test_seq_ids):
#         raise ValueError("Length of test_seq_ids does not match the number of Protein_IDs in the CSV.")

#     # 新增 UniProt_ID 列
#     df['UniProt_ID'] = test_seq_ids

#     # 保存更新后的 DataFrame 到 CSV
#     output_csv_path = 'final_protein_scores_with_uniprot.csv'
#     df.to_csv(output_csv_path, index=False)

#     print("Added UniProt_ID column successfully!")
