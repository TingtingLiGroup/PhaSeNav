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

