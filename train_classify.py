import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from data_prepare import load_train_set
import esm
from dictionary import AutoEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, average_precision_score
import os
from datetime import datetime
from model_mlp import MLPModel, AttentionMLPModel, MultilabelMLPModel,MLPWithL1FeatureSelection
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from typing import Dict, List, Any, Optional,Tuple
import pandas as pd
import copy
from tqdm import tqdm
import gc
import re


# 4. 性能评估
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
def calculate_metrics_binary(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob)
    }

def calculate_metrics_multilabel(y_true, y_pred, y_scores):
    metrics = {}
    num_classes = y_true.shape[1]
    
    for class_idx in range(num_classes):
        metrics[class_idx] = {
            "auc": roc_auc_score(y_true[:, class_idx], y_scores[:, class_idx]),
            "prauc": average_precision_score(y_true[:, class_idx], y_scores[:, class_idx])
        }
    
    return metrics


def train_l1_linear_with_cv_binary_singleclass(
    sae_repr: torch.Tensor,
    label_matrix: torch.Tensor,
    current_class_idx: int,
    n_splits: int = 5,
    epochs: int = 20,
    prev_model_state: Optional[Dict[str, torch.Tensor]] = None,  # 修正类型
    lr: float = 0.001,
    batch_size: int = 64,
    patience: int = 5,
    base_lambda_l1: float = 0.01,
    current_iter: int = 0,  # 新增：当前迭代轮次
    save_dir: str = './binary_l1_models'
) -> Dict[str, Any]:
    os.makedirs(save_dir, exist_ok=True)
    
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    global_blended_importance = None

    # 动态调整L1系数（随着特征减少逐步加强约束）
    lambda_l1 = base_lambda_l1 * (1 + current_iter * 0.2)

    # 继承模型或新建
    def init_model():
        model = MLPWithL1FeatureSelection(
            input_size=sae_repr.shape[1],
            hidden_size1=512,
            hidden_size2=512,
            output_size=1
        ).to(device)
        
        if prev_model_state:
            # 更安全的模型状态加载
            new_state_dict = model.state_dict()
            for k, v in prev_model_state.items():
                if k in new_state_dict and v.shape == new_state_dict[k].shape:
                    new_state_dict[k] = v
            model.load_state_dict(new_state_dict)
            
            # 重置最后一层
            nn.init.xavier_normal_(model.fc3.weight)
            if model.fc3.bias is not None:
                nn.init.zeros_(model.fc3.bias)
        
        return model
        
    # 正类样本
    pos_samples = label_matrix[:, current_class_idx] == 1
    
    # 保存当前类别与其他类别的二分类结果
    class_binary_results = {}
    def custom_loss(outputs, targets, model):
        nonlocal global_blended_importance
        bce_loss = nn.BCELoss()(outputs, targets)
        
        # 直接使用当前模型的特征重要性
        current_importance = model.get_feature_importance()
        global_blended_importance = current_importance.detach().clone()
        
        # L1正则化项
        l1_penalty = lambda_l1 * torch.norm(current_importance.to(device), 1)
        return bce_loss + l1_penalty, current_importance
    
    # 创建当前类别的模型保存目录
    # class_model_dir = os.path.join(save_dir, f'class_{current_class_idx}')
    os.makedirs(save_dir, exist_ok=True)
    
    # 存储所有AUC值
    all_auc_scores = []
    
    # 遍历其他类别作为负样本
    shared_model = init_model()
    for neg_class_idx in range(label_matrix.shape[1]):
        if neg_class_idx == current_class_idx:
            continue
        
        print(f"  二分类: Class {current_class_idx} vs Class {neg_class_idx}")
        
        # # 选择负样本
        # neg_samples = label_matrix[:, neg_class_idx] == 1
        
        # # 创建二分类数据集
        # X_binary = sae_repr[pos_samples | neg_samples]
        # y_binary = torch.zeros_like(X_binary[:, 0], dtype=torch.float32)
        # y_binary[pos_samples[pos_samples | neg_samples]] = 1.0

        # 选择负样本
        neg_samples = label_matrix[:, neg_class_idx] == 1

        # --- 新增: 剔除同时为正负类的样本 ---
        conflict_mask = pos_samples & neg_samples
        if conflict_mask.any():
            print(f"⚠️  Skipping {conflict_mask.sum().item()} overlapping samples for Class {current_class_idx} vs {neg_class_idx}")
            pos_samples_clean = pos_samples & ~conflict_mask
            neg_samples_clean = neg_samples & ~conflict_mask
        else:
            pos_samples_clean = pos_samples
            neg_samples_clean = neg_samples

        # 合并清洗后的样本
        binary_mask = pos_samples_clean | neg_samples_clean
        X_binary = sae_repr[binary_mask]

        # 构建标签
        y_binary = torch.zeros(X_binary.shape[0], dtype=torch.float32)
        y_binary[pos_samples_clean[binary_mask]] = 1.0
        # ✅ 加在这里
        if y_binary.sum() < 2 or (y_binary == 0).sum() < 2:
            print(f"⚠️  Skip Class {current_class_idx} vs {neg_class_idx} due to insufficient data.")
            continue        
        # 检查类别平衡性
        pos_ratio = y_binary.mean().item()
        print(f"  Positive ratio: {pos_ratio:.2%}")
        
        # 交叉验证
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        fold_metrics = []
        
        # 用于存储每个fold的最佳模型
        best_fold_model = None
        best_fold_auc = 0
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_binary)):
            # model = init_model()
            # 数据准备
            X_train, X_val = X_binary[train_idx], X_binary[val_idx]
            y_train, y_val = y_binary[train_idx], y_binary[val_idx]
            
            # 确保标签是二维的
            y_train = y_train.view(-1, 1)
            y_val = y_val.view(-1, 1)
            
            # 创建DataLoader
            train_loader = DataLoader(
                TensorDataset(X_train, y_train),
                batch_size=batch_size,
                shuffle=True
            )
            
            # 模型初始化
            model = MLPWithL1FeatureSelection(
                input_size=X_train.shape[1],
                hidden_size1=512,
                hidden_size2=512,
                output_size=1
            ).to(device)
            
            # 优化器和学习率调度器
            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=2
            )
            
            # 训练循环
            best_val_loss = float('inf')
            early_stop_counter = 0
            best_model_state = None
            
            for epoch in range(epochs):
                model.train()
                total_loss = 0
                
                for X_batch, y_batch in train_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    
                    # 梯度清零
                    optimizer.zero_grad()
                    
                    # 前向传播
                    outputs = model(X_batch)
                    loss, _ = custom_loss(outputs, y_batch, model)
                    
                    # 反向传播
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    
                    total_loss += loss.item()
                
                # 验证阶段
                model.eval()
                with torch.no_grad():
                    X_val_gpu = X_val.to(device)
                    y_val_gpu = y_val.to(device)
                    
                    val_outputs = model(X_val_gpu)
                    val_loss = nn.BCELoss()(val_outputs, y_val_gpu)
                    val_preds = (val_outputs > 0.5).float()
                    
                    # 计算指标
                    # metrics = calculate_metrics_binary(
                    #     y_val_gpu.cpu().numpy(),
                    #     val_preds.cpu().numpy(),
                    #     val_outputs.cpu().numpy()
                    # )
                    # metrics["val_loss"] = val_loss.item()
                    y_true_np = y_val_gpu.cpu().numpy()
                    y_prob_np = val_outputs.cpu().numpy()
                    y_pred_np = val_preds.cpu().numpy()

                    # ✅ 如果验证集中只有一类，跳过该 fold
                    if len(np.unique(y_true_np)) < 2:
                        print(f"⚠️  Skip fold {fold} due to only one class present in validation set.")
                        continue
                    else:
                        metrics = calculate_metrics_binary(y_true_np, y_pred_np, y_prob_np)
                        metrics["val_loss"] = val_loss.item()

                    # 保存每个fold的指标
                    fold_metrics.append(metrics)

                
                # 学习率调整
                scheduler.step(val_loss)
                
                # 早停逻辑
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    early_stop_counter = 0
                    best_model_state = model.state_dict().copy()
                else:
                    early_stop_counter += 1
                    if early_stop_counter >= patience:
                        print(f"Early stopping at epoch {epoch + 1}")
                        break
            
            if best_model_state is not None:
                model.load_state_dict(best_model_state)
            
            # 如果当前fold的AUC最高，则保存模型
            if metrics['roc_auc'] > best_fold_auc:
                best_fold_auc = metrics['roc_auc']
                best_fold_model = copy.deepcopy(model).cpu()

        if len(fold_metrics) == 0:
            print(f"⚠️  All folds skipped for Class {current_class_idx} vs {neg_class_idx}.")
            continue
        # 计算当前二分类任务的平均指标
        avg_metrics = {
            k: np.mean([m[k] for m in fold_metrics]) 
            for k in fold_metrics[0].keys()
        }
        
        # 保存最佳模型
        model_filename = f'class_{current_class_idx}_vs_{neg_class_idx}_best_model.pth'
        model_path = os.path.join(save_dir, model_filename)
        torch.save({
            'state_dict': best_fold_model.state_dict(),  # 使用标准键名
            'architecture': {
                'input_size': X_train.shape[1],
                'hidden_size1': 512,
                'hidden_size2': 512,
                'output_size': 1
            },
            'metadata': {
                'class_pair': (current_class_idx, neg_class_idx),
                'timestamp': datetime.now().isoformat()
            }
        }, model_path)
        
        # 记录当前二分类任务的结果
        current_result = {
            "negative_class": neg_class_idx,
            "model_path": model_path,
            "model_state": best_fold_model.state_dict(),
            "iteration": current_iter,  # ✅ 添加迭代轮次信息
            **avg_metrics
        }
        # class_binary_results.append(current_result)
        class_binary_results[(current_class_idx, neg_class_idx)] = current_result
        # 记录AUC值
        all_auc_scores.append(avg_metrics['roc_auc'])

        if best_fold_model is not None:
            shared_model.load_state_dict(best_fold_model.state_dict())
    
    # 计算17次分类的AUC均值
    mean_auc = np.mean(all_auc_scores)
    std_auc = np.std(all_auc_scores)
    
    return {
        "binary_results": class_binary_results,
        "mean_auc": mean_auc,
        "std_auc": std_auc,
        "all_auc_scores": all_auc_scores,
        "blended_feature_importance": global_blended_importance,  # 用于下一轮
        "model_state": shared_model.state_dict()  # 完整的模型状态
    }


def train_l1_linear_with_cv_binary_all(
    sae_repr: torch.Tensor,
    label_matrix: torch.Tensor,
    current_class_idx: int,
    n_splits: int = 3,
    epochs: int = 20,
    prev_model_state: Optional[Dict[str, torch.Tensor]] = None,
    lr: float = 0.001,
    batch_size: int = 64,
    patience: int = 5,
    base_lambda_l1: float = 0.01,
    current_iter: int = 0,
    save_dir: str = './binary_l1_models'
) -> Dict[str, Any]:
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    global_blended_importance = None
    lambda_l1 = base_lambda_l1 * (1 + current_iter * 0.2)

    def init_model():
        model = MLPWithL1FeatureSelection(
            input_size=sae_repr.shape[1],
            hidden_size1=512,
            hidden_size2=512,
            output_size=1
        ).to(device)

        if prev_model_state:
            new_state_dict = model.state_dict()
            for k, v in prev_model_state.items():
                if k in new_state_dict and v.shape == new_state_dict[k].shape:
                    new_state_dict[k] = v
            model.load_state_dict(new_state_dict)
            nn.init.xavier_normal_(model.fc3.weight)
            if model.fc3.bias is not None:
                nn.init.zeros_(model.fc3.bias)

        return model

    # 创建正负样本
    pos_samples = label_matrix[:, current_class_idx] == 1
    neg_samples = ~pos_samples
    print(f"Binary classification: Class {current_class_idx} vs Rest")
    num_pos = pos_samples.sum().item()
    num_neg = neg_samples.sum().item()
    print(f"Number of positive samples: {num_pos}, Number of negative samples: {num_neg}")

    # 构建数据
    X_binary = sae_repr[pos_samples | neg_samples]
    y_binary = torch.zeros(X_binary.shape[0], dtype=torch.float32)
    y_binary[pos_samples[pos_samples | neg_samples]] = 1.0

    pos_ratio = y_binary.mean().item()
    print(f"  Positive ratio: {pos_ratio:.2%}")
    
    def custom_loss(outputs, targets, model):
        bce_loss = nn.BCELoss()(outputs, targets)
        current_importance = model.get_feature_importance()
        nonlocal global_blended_importance
        global_blended_importance = current_importance.detach().clone()
        l1_penalty = lambda_l1 * torch.norm(current_importance.to(device), 1)
        return bce_loss + l1_penalty, current_importance

    # 交叉验证
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_metrics = []
    best_fold_model = None
    best_fold_auc = 0

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_binary)):
        model = init_model()
        X_train, X_val = X_binary[train_idx], X_binary[val_idx]
        y_train, y_val = y_binary[train_idx], y_binary[val_idx]

        y_train = y_train.view(-1, 1)
        y_val = y_val.view(-1, 1)

        train_loader = DataLoader(
            TensorDataset(X_train, y_train),
            batch_size=batch_size,
            shuffle=True
        )

        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=2
        )

        best_val_loss = float('inf')
        early_stop_counter = 0
        best_model_state = None

        for epoch in range(epochs):
            model.train()
            total_loss = 0

            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss, _ = custom_loss(outputs, y_batch, model)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()

            model.eval()
            with torch.no_grad():
                X_val_gpu = X_val.to(device)
                y_val_gpu = y_val.to(device)
                val_outputs = model(X_val_gpu)
                val_loss = nn.BCELoss()(val_outputs, y_val_gpu)
                val_preds = (val_outputs > 0.5).float()

                metrics = calculate_metrics_binary(
                    y_val_gpu.cpu().numpy(),
                    val_preds.cpu().numpy(),
                    val_outputs.cpu().numpy()
                )
                metrics["val_loss"] = val_loss.item()
            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stop_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                early_stop_counter += 1
                if early_stop_counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break
        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {total_loss/len(train_loader):.4f}")
        if best_model_state is not None:
                model.load_state_dict(best_model_state)

        fold_metrics.append(metrics)

        if metrics['roc_auc'] > best_fold_auc:
            best_fold_auc = metrics['roc_auc']
            best_fold_model = copy.deepcopy(model).cpu()

    
    # 保存最终模型
    avg_metrics = {
        k: np.mean([m[k] for m in fold_metrics])
        for k in fold_metrics[0].keys()
    }

    model_filename = f'class_{current_class_idx}_vs_rest_best_model.pth'
    model_path = os.path.join(save_dir, model_filename)
    torch.save({
        'state_dict': best_fold_model.state_dict(),
        'architecture': {
            'input_size': X_train.shape[1],
            'hidden_size1': 512,
            'hidden_size2': 512,
            'output_size': 1
        },
        'metadata': {
            'class_pair': (current_class_idx, 'rest'),
            'timestamp': datetime.now().isoformat()
        }
    }, model_path)

    class_binary_results = [{
        "negative_class": "rest",
        "model_path": model_path,
        "model_state": best_fold_model.state_dict(),
        "iteration": current_iter,  # 添加迭代轮次信息
        **avg_metrics
    }]

    return {
        "binary_results": class_binary_results,
        "mean_auc": avg_metrics['roc_auc'],
        "std_auc": 0.0,
        "all_auc_scores": [avg_metrics['roc_auc']],
        "blended_feature_importance": global_blended_importance,
        "model_state": best_fold_model.state_dict()
    }
import os
import re  # 仅新增这一行
from tqdm import tqdm
import torch

def compute_sae_repr_from_pt_files(
    pt_folder,
    feature_masks,
    device="cuda:3"
):
    """
    从.pt文件中分批读取稀疏特征，使用掩码做填充、池化，返回池化后的整体张量。

    参数:
        pt_folder: 保存batch_*.pt文件的文件夹路径
        feature_masks: 掩码Tensor，[40960]，bool类型
        device: 运算设备（默认cuda）
    返回:
        sae_repr: [N_proteins, 40960] 的Tensor
    """
    feature_dim = feature_masks.shape[0]
    feature_masks = feature_masks.to(device)

    filtered_features = []

    # -------------------------- 仅修改这部分排序逻辑 --------------------------
    # 获取所有.pt文件路径
    pt_files = [
        os.path.join(pt_folder, fname)
        for fname in os.listdir(pt_folder)
        if fname.endswith(".pt")
    ]
    # 按文件名中的数字从小到大排序（核心改动）
    pt_files = sorted(
        pt_files,
        key=lambda x: int(re.findall(r'\d+', os.path.basename(x))[0])  # 提取数字并排序
    )
    # -------------------------- 排序逻辑结束 --------------------------

    for pt_file in tqdm(pt_files, desc="Processing .pt files"):
        batch_data = torch.load(pt_file, map_location="cpu")

        for protein_id, feature_matrix in batch_data.items():
            feature_matrix = feature_matrix.to(device)  # [L, 40960]

            # 创建掩码特征 [L, 40960]
            masked_feature = torch.zeros(
                feature_matrix.shape[0],
                feature_dim,
                dtype=feature_matrix.dtype,
                device=feature_matrix.device
            )

            masked_feature[:, feature_masks.bool()] = feature_matrix[:, feature_masks.bool()]

            # 平均池化 → [40960]
            sequence_feature = masked_feature.mean(dim=0)
            filtered_features.append(sequence_feature.cpu())  # 回CPU节省显存

            # 清理
            del feature_matrix, masked_feature, sequence_feature

        # del batch_data
        # torch.cuda.empty_cache()
        # gc.collect()

    sae_repr = torch.stack(filtered_features)  # [N, 40960]
    return sae_repr

def train_dynamic_feature_selection(
    # sparse_features: List[torch.Tensor],
    pt_folder: str,
    label_matrix: torch.Tensor,
    num_classes: int = 18,
    feature_dim: int = 40960,
    max_iterations: int = 10,
    feature_threshold: float = 0.005,
    final_feature_num: int = 4000,
    n_splits: int = 5,
    epochs: int = 20,
    lr: float = 0.001,
    base_lambda_l1: float = 0.01,
    batch_size: int = 64,
    class_idx: int = 0,
    patience: int = 7
) -> Dict:
    device_cpu = torch.device('cpu')
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    
    # 初始化全局特征掩码
    feature_masks = torch.ones(feature_dim, dtype=torch.float32, device=device)
    # 计算每轮需要减少的特征数量
    current_feature_num = feature_dim
    features_to_remove_per_iteration = (feature_dim - final_feature_num) // max_iterations
    
    # 性能追踪
    performance_tracking = {
        'iterations': [],
        'class_performances': [],
        'feature_masks': [],
        'feature_importances': []
    }
    
    # 最佳性能记录
    best_performance = {
        'value': float('-inf'),
        'iteration': -1,
        'results': None,
        'feature_masks': None,
        'feature_importances': None
    }
    
    # 连续性能未提升计数器
    performance_stagnant_count = 0
    # 新增继承相关变量
    prev_model_state = None
    
    for iteration in range(max_iterations):
        torch.cuda.empty_cache()
        print(f"\n=== Iteration {iteration + 1}/{max_iterations} ===")
        
        # # 准备特征 - 使用掩码但保持完整特征维度
        # filtered_features = []
        # for feature_matrix in sparse_features:
        #     feature_matrix = feature_matrix.to(device)
        #     # 关键修改：创建全零特征，然后用掩码特征填充
        #     masked_feature = torch.zeros(
        #         feature_matrix.shape[0], feature_dim, 
        #         dtype=feature_matrix.dtype, 
        #         device=feature_matrix.device
        #     )
            
        #     # 仅在掩码为1的位置填充特征
        #     masked_feature[:, feature_masks.bool()] = feature_matrix[:, feature_masks.bool()]
            
        #     # 池化
        #     sequence_feature = masked_feature.mean(dim=0)
        #     filtered_features.append(sequence_feature)
        
        # sae_repr = torch.stack(filtered_features).to(device)
        sae_repr = compute_sae_repr_from_pt_files(
            pt_folder=pt_folder,
            feature_masks=feature_masks,
            device="cuda:3"
        ).to(device)
        # 训练二分类模型
        binary_results = train_l1_linear_with_cv_binary_singleclass(
        # binary_results = train_l1_linear_with_cv_binary_all(
            sae_repr=sae_repr,
            label_matrix=label_matrix.to(device),
            current_class_idx=class_idx,
            n_splits=n_splits,
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            patience=patience,
            base_lambda_l1=base_lambda_l1,
            save_dir=f'./dynamic_feature_models/iteration_{iteration}/class_{class_idx}',
            # 新增继承参数
            prev_model_state=prev_model_state,
            current_iter=iteration,
        )

         # 更新继承参数
        prev_model_state = binary_results["model_state"]
        
        # 计算类别性能
        class_performance = binary_results['mean_auc']
        
        # 根据二分类模型的ROC AUC对特征重要性进行加权
        feature_importance = calculate_weighted_feature_importance(
            binary_results['binary_results'], 
            feature_dim=feature_dim,
            current_mask=feature_masks,  # 传入当前特征掩码
            class_idx = class_idx,
            iterations = iteration
        )
        
         # 计算本轮需要保留的特征数量
        if iteration == max_iterations - 1:
            # 最后一轮直接保留到目标数量
            features_to_keep = final_feature_num
        else:
            # 每轮逐步减少特征数量
            features_to_keep = current_feature_num - features_to_remove_per_iteration
        
        # 选择最重要的特征
        _, top_indices = torch.topk(torch.abs(feature_importance), features_to_keep)
        
        # 更新全局特征掩码
        new_feature_mask = torch.zeros(feature_dim, dtype=torch.float32, device=device)
        new_feature_mask[top_indices] = 1.0
        feature_masks = new_feature_mask
        
        # 性能追踪
        performance_tracking['iterations'].append(iteration)
        performance_tracking['class_performances'].append(class_performance)
        performance_tracking['feature_masks'].append(feature_masks.clone())
        performance_tracking['feature_importances'].append(feature_importance)
        
        # 更新最佳性能
        if class_performance > best_performance['value']:
            best_performance.update({
                'value': class_performance,
                'iteration': iteration,
                'results': binary_results,
                'feature_masks': feature_masks.clone(),
                'feature_importances': feature_importance
            })
            performance_stagnant_count = 0
        else:
            # 即使性能没有提升，也更新特征掩码
            best_performance.update({
                'feature_masks': feature_masks.clone(),
                'feature_importances': feature_importance
            })
            performance_stagnant_count += 1
        
    print("📊 Mean AUC across iterations:")
    for i, auc in enumerate(performance_tracking['class_performances']):
        print(f"  Iteration {i}: {auc:.4f}")
    return {
        'best_iteration': best_performance['iteration'],
        'best_performance': best_performance['value'],
        'best_results': best_performance['results'],
        'best_feature_masks': best_performance['feature_masks'].cpu(),
        'best_feature_importances': best_performance['feature_importances'],
        'performance_tracking': performance_tracking
    }


def calculate_performance(iteration_binary_results):
    """
    计算每个类别17次二分类训练的平均AUC
    
    参数:
    - iteration_binary_results: train_l1_linear_with_cv_binary 的返回结果
    
    返回:
    - 所有类别二分类平均AUC的总体平均值
    """
    # 存储每个类别的平均AUC
    class_avg_aucs = []
    
    # 遍历每个类别的二分类结果
    for class_results in iteration_binary_results.values():
        # 提取当前类别所有二分类任务的AUC
        class_binary_aucs = [
            binary_result['roc_auc'] 
            for binary_result in class_results
        ]
        
        # 计算当前类别的平均AUC
        class_avg_auc = np.mean(class_binary_aucs)
        class_avg_aucs.append(class_avg_auc)
    
    # 计算所有类别平均AUC的总体平均值
    overall_avg_auc = np.mean(class_avg_aucs)
    
    # 可选：打印详细信息
    print("每个类别的平均AUC:")
    for class_idx, avg_auc in enumerate(class_avg_aucs):
        print(f"Class {class_idx}: {avg_auc:.4f}")
    print(f"总体平均AUC: {overall_avg_auc:.4f}")
    
    return overall_avg_auc


def calculate_weighted_feature_importance(
    binary_results: List[Dict], 
    current_mask: torch.Tensor,
    class_idx: int = 0,
    iterations: int = 0,
    feature_dim: int = 40960
) -> torch.Tensor:
    # 初始化特征重要性向量
    total_importance = torch.zeros(feature_dim, dtype=torch.float32)
    total_weight = 0
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    base_path = f"./dynamic_feature_models/iteration_{iterations}/class_{class_idx}/"
    
    # 如果没有传入mask，默认全部特征
    if current_mask is None:
        current_mask = torch.ones(feature_dim, dtype=torch.float32)
    
    # 确保current_mask是张量
    current_mask = torch.as_tensor(current_mask, dtype=torch.float32)
    
    # 获取当前有效特征索引
    current_indices = torch.nonzero(current_mask > 0).squeeze()
    
    # for binary_result in binary_results:
    for binary_result in binary_results.values():
        # 使用ROC AUC作为权重
        weight_performance = binary_result.get('roc_auc', 0.5)
        negative_class = binary_result.get('negative_class')
        
        # 加载模型
        model_path = os.path.join(base_path, f"class_{class_idx}_vs_{negative_class}_best_model.pth")
        
        model, input_size = load_model(model_path, device)
        
        with torch.no_grad():
            # 获取各层权重 
            w1 = model.fc1.weight.detach().cpu().numpy()  # [hidden1, input]
            w2 = model.fc2.weight.detach().cpu().numpy()  # [hidden2, hidden1]
            w3 = model.fc3.weight.detach().cpu().numpy()  # [1, hidden2]
            
            # 计算特征到输出的总贡献（绝对值传播）
            importance = np.abs(w1.T @ w2.T @ w3.T).flatten()
            
            # 创建完整维度的重要性向量
            full_importance = torch.zeros(feature_dim, dtype=torch.float32)
            full_importance[current_indices] = torch.from_numpy(importance[current_indices.cpu().numpy()]).float()
            
            # 加权累积
            total_importance += full_importance * weight_performance
            total_weight += weight_performance
    
    # 归一化
    avg_importance = total_importance / (total_weight + 1e-8)
    
    return avg_importance

def load_model(checkpoint_path, device):
    """
    安全加载模型到指定设备（兼容新旧格式）
    
    参数:
        checkpoint_path: 模型路径
        device: 目标设备
    
    返回:
        model: 加载完成的模型
        input_size: 输入特征维度
    """
    try:
        # 1. 加载检查点
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        # 2. 格式解析
        if isinstance(checkpoint, dict):
            # 新版格式（含architecture信息）
            if 'state_dict' in checkpoint and 'architecture' in checkpoint:
                state_dict = checkpoint['state_dict']
                arch = checkpoint['architecture']
                input_size = arch['input_size']
                hidden_size1 = arch['hidden_size1']
                hidden_size2 = arch['hidden_size2']
            
            # 旧版直接state_dict格式
            elif any(k.startswith(('fc1.', 'hidden_layer1.')) for k in checkpoint.keys()):
                state_dict = checkpoint
                # 自动推断网络结构
                first_layer_key = next(k for k in state_dict.keys() if k.startswith(('fc1.', 'hidden_layer1.')))
                input_size = state_dict[first_layer_key].shape[1]
                hidden_size1 = state_dict[first_layer_key].shape[0]
                second_layer_key = next(k for k in state_dict.keys() if k.startswith(('fc2.', 'hidden_layer2.')))
                hidden_size2 = state_dict[second_layer_key].shape[0]
            
            else:
                raise ValueError("无法识别的字典格式")
        
        # 3. 模型初始化
        model = MLPWithL1FeatureSelection(
            input_size=input_size,
            hidden_size1=hidden_size1,
            hidden_size2=hidden_size2,
            output_size=1
        ).to(device)
        
        # 4. 权重加载（处理可能的键名差异）
        # 移除可能的'module.'前缀（多GPU训练产生）
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        # 非严格模式加载（允许缺失部分参数）
        model.load_state_dict(state_dict, strict=False)
        
        model.eval()
        return model, input_size
    
    except Exception as e:
        print(f"[错误] 加载模型失败: {checkpoint_path}")
        if isinstance(checkpoint, dict):
            print("检查点包含的键:", checkpoint.keys())
            if 'state_dict' in checkpoint:
                print("state_dict中的键示例:", list(checkpoint['state_dict'].keys())[:3])
        raise ValueError(f"模型加载错误: {str(e)}")

def evaluate_on_test_set_mlp(
    # sparse_features_test: List[torch.Tensor],
    pt_folder: str,
    label_matrix_test: torch.Tensor,
    best_feature_mask: torch.Tensor,
    model_state_dict: Dict,
    input_dim: int,
    class_idx: int = 0,
    batch_size: int = 64,
    feature_dim: int = 40960,
    device: str = "cuda:1" if torch.cuda.is_available() else "cpu"
) -> Dict:
    """
    使用训练得到的 MLPWithL1FeatureSelection 模型在测试集上评估性能。
    """
    device = torch.device(device)
    
    # 1. 构建测试集特征（带掩码）
    # filtered_features = []
    # for feature_matrix in sparse_features_test:
    #     masked_feature = torch.zeros(
    #         feature_matrix.shape[0], input_dim,
    #         dtype=feature_matrix.dtype,
    #         device=feature_matrix.device
    #     )
    #     masked_feature[:, best_feature_mask.bool()] = feature_matrix[:, best_feature_mask.bool()]
    #     sequence_feature = masked_feature.mean(dim=0)
    #     filtered_features.append(sequence_feature)

    # sae_repr_test = torch.stack(filtered_features).to(device)
    sae_repr_test = compute_sae_repr_from_pt_files(
    pt_folder=pt_folder,        # 指向你保存的测试集 .pt 文件目录
    feature_masks=best_feature_mask,           # 这里用的是最终选出的掩码
    device="cuda:1"
    ).to(device)


    y_test = label_matrix_test[:, class_idx].float().to(device)

    # 2. 加载模型并加载参数
    model = MLPWithL1FeatureSelection(input_size=input_dim).to(device)
    model.load_state_dict(model_state_dict)
    model.eval()

    # 3. 前向推理
    preds = []
    with torch.no_grad():
        for i in range(0, len(sae_repr_test), batch_size):
            batch = sae_repr_test[i:i+batch_size]
            probs = model(batch).squeeze()
            preds.append(probs)

    all_probs = torch.cat(preds).cpu().numpy()
    true_labels = y_test.cpu().numpy()



    auc = roc_auc_score(true_labels, all_probs)
    ap = average_precision_score(true_labels, all_probs)
    acc = accuracy_score(true_labels, (all_probs > 0.5).astype(int))

    return {
        "AUC": auc,
        "AP": ap,
        "Accuracy": acc,
        "probs": all_probs,
        "true_labels": true_labels
    }

def evaluate_on_test_set_mlp_from_pth(
    model_filepath: str,  # 新增参数：模型文件路径
    pt_folder: str,
    label_matrix_test: torch.Tensor,
    best_feature_mask: torch.Tensor,
    input_dim: int,
    class_idx: int = 0,
    batch_size: int = 64,
    feature_dim: int = 40960,
    device: str = "cuda:1" if torch.cuda.is_available() else "cpu"
) -> Dict:
    """
    使用训练得到的 MLPWithL1FeatureSelection 模型在测试集上评估性能。
    """
    device = torch.device(device)

    # 1. 构建测试集特征（带掩码）
    sae_repr_test = compute_sae_repr_from_pt_files(
        pt_folder=pt_folder,  # 指向你保存的测试集 .pt 文件目录
        feature_masks=best_feature_mask,  # 这里用的是最终选出的掩码
        device=device
    ).to(device)

    y_test = label_matrix_test[:, class_idx].float().to(device)

    # 获取测试集中的正样本数量和总样本数量
    total_samples = y_test.size(0)  # 所有测试样本的数量
    positive_samples = (y_test == 1).sum().item()  # 正样本的数量

    print(f"总样本数量: {total_samples}")
    print(f"正样本数量: {positive_samples}")

    # 2. 加载模型
    model = MLPWithL1FeatureSelection(input_size=input_dim).to(device)
    model.load_state_dict(torch.load(model_filepath)['state_dict'])  # 从文件加载模型状态
    model.eval()

    # 3. 前向推理
    preds = []
    with torch.no_grad():
        for i in range(0, len(sae_repr_test), batch_size):
            batch = sae_repr_test[i:i+batch_size]
            probs = model(batch).squeeze()
            preds.append(probs)
    
    all_probs = torch.cat(preds).cpu().numpy()
    true_labels = y_test.cpu().numpy()

    # 计算性能指标
    auc = roc_auc_score(true_labels, all_probs)
    ap = average_precision_score(true_labels, all_probs)
    acc = accuracy_score(true_labels, (all_probs > 0.5).astype(int))

    return {
        "AUC": auc,
        "AP": ap,
        "Accuracy": acc,
        "probs": all_probs,
        "true_labels": true_labels
    }

def evaluate_1vs1_model_on_test_subset(
    pt_folder: str,
    label_matrix_test: torch.Tensor,
    best_feature_mask: torch.Tensor,
    model_state_dict: Dict,
    input_dim: int,
    class_a: int,
    class_b: int,
    batch_size: int = 64,
    feature_dim: int = 40960,
    device: str = "cuda:1"
) -> Dict:
    """
    用训练好的 1-vs-1 模型在测试集中只对 class_a 和 class_b 的样本进行评估。
    返回 AUC / AP / Accuracy。
    """
    device = torch.device(device)

    # 加载特征并进行掩码池化
    sae_repr_test = compute_sae_repr_from_pt_files(
        pt_folder=pt_folder,
        feature_masks=best_feature_mask,
        device=device
    ).to(device)

    # 找到测试集中属于 class_a 或 class_b 的样本索引
    label_a = label_matrix_test[:, class_a]
    label_b = label_matrix_test[:, class_b]
    selected_indices = ((label_a == 1) ^ (label_b == 1)).nonzero(as_tuple=True)[0]
    print(f"🔍 Selected {len(selected_indices)} test samples for class_{class_a} vs class_{class_b}")


    if len(selected_indices) == 0:
        print(f"⚠️ 测试集中没有 class_{class_a} vs class_{class_b} 的样本！")
        return None

    x_test = sae_repr_test[selected_indices]
    y_test = label_a[selected_indices].float().to(device)  # 将 class_a 作为正类（1），class_b 为负类（0）

    # 加载模型
    model = MLPWithL1FeatureSelection(input_size=input_dim).to(device)
    model.load_state_dict(model_state_dict)
    model.eval()

    # 前向推理
    probs = []
    with torch.no_grad():
        for i in range(0, x_test.shape[0], batch_size):
            batch = x_test[i:i + batch_size]
            pred = model(batch)
            if pred.ndim == 2 and pred.shape[1] == 1:
                pred = pred.squeeze(1)  # 只压缩第1维，保留batch维度
            probs.append(pred)

    probs = torch.cat(probs, dim=0).cpu().numpy()
    y_true = y_test.cpu().numpy()
    y_pred = (probs > 0.5).astype(int)

    auc = roc_auc_score(y_true, probs)
    ap = average_precision_score(y_true, probs)
    acc = accuracy_score(y_true, (probs > 0.5).astype(int))
    f1  = f1_score(y_true, y_pred)

    return {
        "AUC": auc,
        "AP": ap,
        "Accuracy": acc,
        "probs": probs,
        "F1": f1,
        "true_labels": y_true
    }


def evaluate_all_1vs1_models_on_test(
    pt_folder: str,
    label_matrix_test: torch.Tensor,
    best_feature_mask: torch.Tensor,
    binary_results: Dict,
    input_dim: int,
    batch_size: int = 64,
    device: str = "cuda:1"
):
    """
    针对训练好的所有 1-vs-1 模型，在测试集对应的子集上进行测试。
    支持从模型文件路径加载。
    返回一个包含每个pair的性能字典。
    """
    evaluation_summary = {}

    for (target_class, compare_class), result in binary_results.items():
        print(f"\n🔍 Evaluating {target_class} vs {compare_class} ...")

        # ✅ 尝试从model_state中获取，否则从model_path加载
        if "model_state" in result:
            model_state = result["model_state"]
            iteration_info = result.get("iteration", "unknown")
            print(f"✅ Using in-memory model from iteration {iteration_info} for ({target_class}, {compare_class})")
        elif "model_path" in result:
            print(f"📂 Loading model from {result['model_path']}")
            checkpoint = torch.load(result["model_path"], map_location=device)
            model_state = checkpoint["state_dict"]
        else:
            print(f"❌ No model state or path found for ({target_class}, {compare_class})")
            continue

        eval_result = evaluate_1vs1_model_on_test_subset(
            pt_folder=pt_folder,
            label_matrix_test=label_matrix_test,
            best_feature_mask=best_feature_mask,
            model_state_dict=model_state,
            input_dim=input_dim,
            class_a=target_class,
            class_b=compare_class,
            batch_size=batch_size,
            device=device
        )

        if eval_result is not None:
            evaluation_summary[(target_class, compare_class)] = {
                "AUC": eval_result["AUC"],
                "AP": eval_result["AP"],
                "Accuracy": eval_result["Accuracy"],
            }

    return evaluation_summary

def evaluate_all_1vs1_models_on_test_from_pth(
    pt_folder: str,
    label_matrix_test: torch.Tensor,
    best_feature_mask: torch.Tensor,
    model_folder: str,
    input_dim: int,
    batch_size: int = 64,
    device: str = "cuda:1"
):
    evaluation_summary = {}
    auc_scores = []  # 新增：存储所有模型的AUC
    f1_scores  = []  # ← 新增
    ap_scores  = []
    
    # 获取模型文件夹中的所有模型文件
    model_files = sorted(
        [f for f in os.listdir(model_folder) if f.endswith('.pth')],
        key=lambda x: (
            int(x.split('_')[1]),  # 按第一个类别索引排序
            int(x.split('_')[3].split('.')[0])  # 按第二个类别索引排序
        )
    )
    
    for model_file in model_files:
        try:
            # 直接从文件名解析类别索引
            parts = model_file[:-4].split('_')
            class_a = int(parts[1])  # 直接使用数字索引
            class_b = int(parts[3])  # 直接使用数字索引
            
            model_path = os.path.join(model_folder, model_file)
            checkpoint = torch.load(model_path, map_location=device)
            model_state = checkpoint['state_dict']
            
            print(f"\n🔍 Evaluating class {class_a} vs class {class_b} ...")
        
        except (ValueError, IndexError) as e:
            print(f"❌ 无法解析模型文件名: {model_file}, 错误: {e}")
            continue
        
        # 评估模型
        eval_result = evaluate_1vs1_model_on_test_subset(
            pt_folder=pt_folder,
            label_matrix_test=label_matrix_test,
            best_feature_mask=best_feature_mask,
            model_state_dict=model_state,
            input_dim=input_dim,
            class_a=class_a,
            class_b=class_b,
            batch_size=batch_size,
            device=device
        )
        
        if eval_result is not None:
            evaluation_summary[(class_a, class_b)] = {
                "AUC":      eval_result["AUC"],
                "AP":       eval_result["AP"],
                "Accuracy": eval_result["Accuracy"],
                "F1":       eval_result["F1"],   # ← 新增
            }
            auc_scores.append(eval_result["AUC"])
            f1_scores.append(eval_result["F1"])  # ← 新增
            ap_scores.append(eval_result["AP"])
    
    if auc_scores:
        avg_auc = np.mean(auc_scores)
        std_auc = np.std(auc_scores)
        avg_f1  = np.mean(f1_scores)             # ← 新增
        std_f1  = np.std(f1_scores)              # ← 新增
        avg_ap  = np.mean(ap_scores)          # ← 新增
        std_ap  = np.std(ap_scores)           # ← 新增
        
        print("\n📊 总体模型性能汇总:")
        print(f"平均 AUC: {avg_auc:.4f} ± {std_auc:.4f}")
        print(f"平均 AP:  {avg_ap:.4f} ± {std_ap:.4f}") 
        print(f"平均 F1:  {avg_f1:.4f} ± {std_f1:.4f}")  # ← 新增
        
        return {
            "evaluation_summary": evaluation_summary,
            "avg_auc": avg_auc,
            "std_auc": std_auc,
            "avg_ap":  avg_ap,   # ← 新增
            "std_ap":  std_ap,   # ← 新增
            "avg_f1":  avg_f1,
            "std_f1":  std_f1,
        }
    
    return evaluation_summary

def evaluate_1vs1_model_on_test_subset_to_new_prots(
    pt_folder: str,
    best_feature_mask: torch.Tensor,
    model_path: str,
    input_dim: int,
    class_a: int,
    class_b: int,
    batch_size: int = 64,
    device: str = "cuda:1"
) -> np.ndarray:
    """
    用训练好的 1-vs-1 模型在测试集中只对 class_a 和 class_b 的样本进行评估。
    返回预测概率。
    """
    device = torch.device(device)
    
    # 加载特征并进行掩码池化
    sae_repr_test = compute_sae_repr_from_pt_files(
        pt_folder=pt_folder,
        feature_masks=best_feature_mask,
        device=device
    ).to(device)

    # 加载模型
    model = MLPWithL1FeatureSelection(input_size=input_dim).to(device)
    model.load_state_dict(torch.load(model_path)['state_dict'])  # 从文件加载模型状态
    model.eval()

    # 前向推理
    probs = []
    with torch.no_grad():
        for i in range(0, sae_repr_test.shape[0], batch_size):
            batch = sae_repr_test[i:i + batch_size]
            pred = model(batch)
            if pred.ndim == 2 and pred.shape[1] == 1:
                pred = pred.squeeze(1)  # 只压缩第1维，保留batch维度
            probs.append(pred)

    probs = torch.cat(probs, dim=0).cpu().numpy()
    return probs  # 只返回预测概率

def evaluate_all_1vs1_models_on_test_to_new_prots(
    model_filepath: str,
    pt_folder: str,
    best_feature_mask: torch.Tensor,
    input_dim: int,
    train_auc: Dict[str, float],  # 直接传入 train_auc 字典
    batch_size: int = 64,
    device: str = "cuda:1"
) -> Optional[List[Tuple[str, float]]]:  # 可以返回 None
    """
    针对训练好的所有 1-vs-1 模型，在测试集对应的子集上进行测试。
    收集每个模型的概率，并根据 AUC 加权最终得分。
    返回每个样本的最终得分。
    """
    final_scores = None  # 用于保存加权得分
    protein_scores = []  # 用于保存每个蛋白质的最终得分和对应的蛋白质标识符

    # 获取模型文件夹中的所有模型文件
    # 假设传入的 model_filepath 是形如 "./dynamic_feature_models/iteration_8/class_4"
    for compare_class in range(14):  # 假设 compare_class 从 0 到 13
        target_class = 4  # 此处使用替代目标类
        if compare_class == 4:  # 跳过与自己比较
            continue
        model_file = f'class_4_vs_{compare_class}_best_model.pth'
        model_path = os.path.join(model_filepath, model_file)
        
        if not os.path.exists(model_path):
            print(f"❌ Model file does not exist: {model_path}")
            continue

        print(f"\n🔍 Evaluating class_4 vs class_{compare_class} ...")

        # 获取模型预测概率
        # 注意这里 class_a 是目标类，应始终为 4
        
        probs = evaluate_1vs1_model_on_test_subset_to_new_prots(
            pt_folder=pt_folder,
            best_feature_mask=best_feature_mask,
            model_path=model_path,
            input_dim=input_dim,
            class_a=target_class,
            class_b=compare_class,
            batch_size=batch_size,
            device=device
        )

        # 确保 `probs` 有有效结果
        if probs is None or len(probs) == 0:
            print(f"❌ No predictions found for model class_{target_class} vs class_{compare_class}")
            continue
        
        # 计算 AUC 以用于加权
        if f"{target_class}_vs_{compare_class}" in train_auc:
            auc = train_auc[f"{target_class}_vs_{compare_class}"]
        else:
            print(f"❌ AUC not found for model ({target_class}, {compare_class})")
            continue
        
        # 初始化 final_scores
        if final_scores is None:
            final_scores = probs * auc
        else:
            final_scores += probs * auc

    # 如果没有有效的模型评分，返回 None
    if final_scores is None:
        print("❌ No valid scores calculated. Check if any models were successfully evaluated.")
        return None

    # 标准化最终得分
    final_scores /= sum(train_auc[f"{target_class}_vs_{compare_class}"] for compare_class in range(14) if f"{target_class}_vs_{compare_class}" in train_auc)

    # 将最终得分与蛋白质匹配
    protein_ids = [f'Protein_{i}' for i in range(len(final_scores))]  # 这里假设蛋白质ID的生成
    protein_scores = [(protein_ids[i], final_scores[i]) for i in range(len(final_scores))]

    return protein_scores  # 返回包含蛋白质ID及其得分的元组列表

def train_mlp_with_cv_multilabel(
    sae_repr: torch.Tensor,
    label_matrix: torch.Tensor,
    num_classes: int = 18,
    n_splits: int = 3,
    epochs: int = 20,
    lr: float = 0.001,
    batch_size: int = 64,
    patience: int = 5
) -> Dict[int, Dict[str, Any]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}
    
    # Prepare for KFold cross-validation
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(sae_repr)):
        print(f"\n=== Training Fold {fold + 1}/{n_splits} ===")
        
        # 分割训练集和验证集
        X_train, X_val = sae_repr[train_idx], sae_repr[val_idx]
        y_train, y_val = label_matrix[train_idx], label_matrix[val_idx]
        
        # 创建 DataLoader
        train_dataset = TensorDataset(X_train.float(), y_train.float())
        val_dataset = TensorDataset(X_val.float(), y_val.float())
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # 模型初始化
        model = MultilabelMLPModel(
            input_size=X_train.shape[1],  # 特征维度
            hidden_size1=512,
            hidden_size2=256,
            num_classes=num_classes
        ).to(device)
        
        # 损失函数和优化器
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=2
        )
        
        # 训练循环
        best_val_loss = float('inf')
        early_stop_counter = 0
        best_model_state = None
        
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                
                # 清空梯度
                optimizer.zero_grad()
                
                # 前向传播
                outputs = model(X_batch)
                
                # 计算损失
                loss = criterion(outputs, y_batch)
                
                # 反向传播
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # 添加训练损失输出
            print(f"Epoch {epoch+1}, Train Loss: {train_loss/len(train_loader):.4f}")
            # 验证阶段
            model.eval()
            val_losses = []
            all_preds = []
            all_labels = []
            
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(device)
                    y_batch = y_batch.to(device)
                    
                    # 前向传播
                    outputs = model(X_batch)
                    
                    # 计算验证损失
                    val_loss = criterion(outputs, y_batch)
                    val_losses.append(val_loss.item())
                    
                    # 计算预测概率
                    preds = torch.sigmoid(outputs)
                    all_preds.append(preds.cpu().numpy())
                    all_labels.append(y_batch.cpu().numpy())
                
                # 添加验证损失输出
                print(f"Validation Loss: {np.mean(val_losses):.4f}")
                # 合并预测结果和标签
                all_preds = np.concatenate(all_preds, axis=0)
                all_labels = np.concatenate(all_labels, axis=0)
                
                # 计算指标
                metrics = calculate_metrics_multilabel(
                    all_labels, 
                    (all_preds > 0.5).astype(float), 
                    all_preds
                )
                
                val_loss = np.mean(val_losses)
                metrics['val_loss'] = val_loss
                
                # 学习率调整
                scheduler.step(val_loss)
                
                # 早停
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    early_stop_counter = 0
                    best_model_state = model.state_dict().copy()
                else:
                    early_stop_counter += 1
                    if early_stop_counter >= patience:
                        print(f"Early stopping at epoch {epoch + 1}")
                        break
        
        # 保存每个折的结果
        results[fold] = {
            "best_val_loss": best_val_loss,
            "metrics": metrics,
            "best_model_state": best_model_state
        }
        
        # 保存最佳模型
        if best_model_state is not None:
            torch.save(best_model_state, f"best_model_fold_{fold}.pt")
    
    return results