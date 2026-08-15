import sys
import os
import importlib.util
import torch
import numpy as np
from typing import Tuple, Optional, List, Dict
import py3Dmol
import pandas as pd
from tqdm import tqdm
import esm
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForTokenClassification
import random


# ===================== 核心配置：路径定义（解决所有导入冲突） =====================
# 获取当前脚本绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 3. 自定义包（dictionary）路径
dictionary_dir = os.path.join(current_dir, os.pardir)
dictionary_dir = os.path.abspath(dictionary_dir)
model_mlp_dir = os.path.join(current_dir, os.pardir)
model_mlp_dir = os.path.abspath(dictionary_dir)
# 添加自定义包路径
if dictionary_dir not in sys.path:
    sys.path.append(dictionary_dir)
from dictionary import AutoEncoder
from model_mlp import MLPWithL1FeatureSelection


# ===================== 关键修复：添加ESM3外层根目录（不是内层） =====================
esm3_root = os.path.join(current_dir, "esm3")  # 外层根目录（和手动命令一致）
sys.path.insert(0, esm3_root)  # 仅改这一行！

from esm3.models.esm3 import ESM3
from esm3.sdk.api import ESMProtein, GenerationConfig
from esm3.sdk.experimental import ESM3GuidedDecoding, GuidedDecodingScoringFunction

# ======== DR-BERT单例加载及联合评分函数开始 ========
DRBERT_MODEL_DIR = "./DR-BERT/checkpoint-final"  # 请根据实际路径修改
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

class DRBERTScorer:
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        print("🟢 加载 DR-BERT 模型和tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(DRBERT_MODEL_DIR)
        self.model = AutoModelForTokenClassification.from_pretrained(DRBERT_MODEL_DIR)
        self.model.eval()
        self.model.to(DEVICE)
        print(f"🟢 DR-BERT 加载完成，使用设备：{DEVICE}")
        self._initialized = True
    
    def get_drbert_idr_score(self, sequence: str) -> float:
        if not sequence:
            return 0.0
        seq_trunc = sequence[:1022]
        encoded = self.tokenizer.encode_plus(seq_trunc, return_tensors="pt")
        encoded = {k: v.to(DEVICE) for k,v in encoded.items()}
        with torch.no_grad():
            output = self.model(**encoded)
        logits = output['logits'].squeeze(0)  # (seq_len, 2)
        probs = F.softmax(logits, dim=-1)[:,1]
        idr_probs = probs[1:-1]
        if len(idr_probs) == 0:
            return 0.0
        return idr_probs.mean().item()
    
    def combined_score(self, sequence: str, idr_weight: float = 0.3) -> float:
        classifier_score = ProteinEvaluator.score_final_sequence(sequence)
        idr_score = self.get_drbert_idr_score(sequence)
        return (1 - idr_weight)*classifier_score + idr_weight*idr_score

# 单例实例，供其他位置调用
drbert_scorer = DRBERTScorer()
def combined_sequence_score(sequence: str, idr_weight=None) -> float:
    if idr_weight is None:
        idr_weight = Config.IDR_WEIGHT
    return drbert_scorer.combined_score(sequence, idr_weight)

# ===================== 新增：氨基酸字符集过滤工具 =====================
def filter_amino_acid_sequence(sequence: str) -> str:
    """
    过滤非标准氨基酸，仅保留20种标准氨基酸（替换非标准字符为最常见的G/随机标准氨基酸）
    参数:
        sequence: 原始生成的序列（可能包含X/B/J/U/_等字符）
    返回:
        过滤后的纯标准氨基酸序列
    """
    # 20种标准氨基酸的单字母缩写（天然氨基酸）
    STANDARD_AMINO_ACIDS = {'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'}
    # 可选：替换非标准字符的默认氨基酸（G是柔性残基，符合你的无序序列需求）
    DEFAULT_AA = 'G'

    filtered_seq = []
    for char in sequence:
        # 保留标准氨基酸，其余替换为DEFAULT_AA（可改为随机选标准氨基酸）
        if char in STANDARD_AMINO_ACIDS:
            filtered_seq.append(char)
        else:
            filtered_seq.append(DEFAULT_AA)
    return ''.join(filtered_seq)
# ================================ 配置模块 ================================
class Config:
    """配置参数类（移除结构相关配置）"""
    # 模型配置
    MODEL_NAME = "esm3-sm-open-v1"
    INFRA_PROVIDER = "True"
    TARGET_CLASS = 4  # 目标预测类别
    
    # 生成参数（重点调整：提高温度增强无序性）
    PROTEIN_LENGTH = 100             # 肽段长度
    NUM_DECODING_STEPS = 25          # 解码步数
    NUM_SAMPLES_PER_STEP = 10        # 每步候选数
    # 提高去噪温度（α增大至0.7~1.0）：增强序列多样性，降低结构倾向性
    DENOISED_PREDICTION_TEMP = 0.0   
    # 无引导生成温度提高至1.0：鼓励生成更无序的序列
    UNGUIDED_TEMP = 0.0 
    # 批量处理配置
    MASK_START_POSITION = 20            
    IDR_WEIGHT = 0.5
    # 保留特征相关配置（用于打分）
    ESM2_MODEL_NAME = "esm2_t33_650M_UR50D"
    SAE_CHK_PATH = os.path.join(dictionary_dir, "step_80000.pt")
    CLASSIFIER_MODEL_PATH = os.path.join(dictionary_dir, f"dynamic_feature_models/iteration_4/class_{TARGET_CLASS}")
    FEATURE_CSV_PATH = os.path.join(dictionary_dir, f"final_results_20250724/selected_features_{TARGET_CLASS}.csv")
    AUC_CSV_PATH = os.path.join(dictionary_dir, f"final_results_20250724/best_model_performance_{TARGET_CLASS}.csv")
    FEATURE_DIM = 40960
    DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# ================================ 加载核心模型（全局初始化，避免重复加载） ================================
class CoreModelLoader:
    """加载ESM2、SAE、分类器、特征掩码等核心组件（单例模式，避免重复加载）"""
    _instance = None  # 单例实例
    _loaded = False   # 加载状态标记（避免重复加载）
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            # 初始化属性
            cls._instance.esm2_model = None
            cls._instance.esm2_alphabet = None
            cls._instance.esm2_batch_converter = None
            cls._instance.sae_model = None
            cls._instance.binary_mask = None
            cls._instance.train_auc = None
            cls._instance.classifiers = {}  # 缓存预加载的分类器
        return cls._instance
    
    def load_all(self):
        if self._loaded:
            return self
        
        print("🔧 正在加载核心模型（ESM2 + SAE + 分类器）...")
        
        # 1. 加载ESM2模型
        try:
            self.esm2_model, self.esm2_alphabet = esm.pretrained.esm2_t33_650M_UR50D()
            self.esm2_batch_converter = self.esm2_alphabet.get_batch_converter()
            self.esm2_model.eval()
            self.esm2_model = self.esm2_model.to(Config.DEVICE)
        except AttributeError as e:
            raise RuntimeError(f"ESM2模型加载失败：{str(e)}") from e
        
        # 2. 加载SAE模型
        try:
            self.sae_model = AutoEncoder.from_pretrained(Config.SAE_CHK_PATH)
            self.sae_model.eval().to(Config.DEVICE)
        except Exception as e:
            raise RuntimeError(f"SAE模型加载失败：{str(e)}") from e
        
        # 3. 加载特征掩码
        try:
            df = pd.read_csv(Config.FEATURE_CSV_PATH)
            feature_indices = df['feature_index'].astype(int).tolist()
            self.binary_mask = torch.zeros(Config.FEATURE_DIM, dtype=torch.float32).to(Config.DEVICE)
            for idx in feature_indices:
                if 0 <= idx < Config.FEATURE_DIM:
                    self.binary_mask[idx] = 1.0
        except Exception as e:
            raise RuntimeError(f"特征掩码加载失败：{str(e)}") from e
        
        # 4. 加载AUC结果
        try:
            auc_data = pd.read_csv(Config.AUC_CSV_PATH)
            self.train_auc = auc_data.set_index('class_pair').to_dict()['roc_auc']
        except Exception as e:
            raise RuntimeError(f"AUC结果加载失败：{str(e)}") from e
        
        # 5. 预加载分类器
        target_class = Config.TARGET_CLASS
        model_path = Config.CLASSIFIER_MODEL_PATH
        
        for compare_class in range(14):
            if compare_class == target_class:
                continue
            
            model_key = f"class_{target_class}_vs_{compare_class}"
            model_file = f"{model_key}_best_model.pth"
            full_path = os.path.join(model_path, model_file)
            
            if not os.path.exists(full_path):
                continue
            
            try:
                model = MLPWithL1FeatureSelection(input_size=Config.FEATURE_DIM).to(Config.DEVICE)
                model.load_state_dict(torch.load(full_path, map_location=Config.DEVICE)['state_dict'])
                model.eval()
                self.classifiers[model_key] = model
            except Exception as e:
                continue
        
        # 统一打印加载结果
        print(f"✅ 核心模型加载完成：ESM2 | SAE | 分类器({len(self.classifiers)}/13)")
        self._loaded = True
        return self

# ================================ 评分函数（适配 ESM3 内置接口） ================================
class CustomClassifierScoringFunction(GuidedDecodingScoringFunction):
    """
    修改后兼顾传统分类器评分和DR-BERT无序预测的联合评分，用于ESM3引导生成内的评分。
    """
    def __init__(self, idr_weight=0.3):
        super().__init__()

        # 核心模型加载（保持你已有逻辑）
        self.core_loader = CoreModelLoader().load_all()
        self.esm2_model = self.core_loader.esm2_model
        self.esm2_alphabet = self.core_loader.esm2_alphabet
        self.esm2_batch_converter = self.core_loader.esm2_batch_converter
        self.sae_model = self.core_loader.sae_model
        self.binary_mask = self.core_loader.binary_mask
        self.train_auc = self.core_loader.train_auc
        self.classifiers = self.core_loader.classifiers
        self.device = Config.DEVICE
        self.target_class = Config.TARGET_CLASS

        # DR-BERT模型加载和tokenizer（单例）
        self.idr_weight = idr_weight if idr_weight is not None else Config.IDR_WEIGHT
        self.drbert_scorer = drbert_scorer


    def __call__(self, protein: ESMProtein) -> float:
        seq = protein.sequence

        # 允许含掩码序列，且长度限制
        if len(seq) < 10 or len(seq) > 1024:
            return 0.0

        try:
            # 先计算单分类器score
            classifier_score = self._compute_classifier_score(seq)
            # 再计算idr_score
            idr_score = self._get_drbert_idr_score(seq)
            # 联合
            final_score = (1 - self.idr_weight) * classifier_score + self.idr_weight * idr_score

            # 确保float，不是tensor
            if isinstance(final_score, torch.Tensor):
                final_score = final_score.item()
            return final_score

        except Exception as e:
            print(f"⚠️  引导评分函数调用失败：{str(e)}")
            return 0.0

    def _compute_classifier_score(self, seq: str) -> float:
        # 复用原有核心逻辑
        residue_vector = self._sequence_to_esm2_residue_vector(seq)
        if residue_vector is None:
            return 0.0
        sparse_vector = self._esm2_to_sparse_vector(residue_vector)
        if sparse_vector is None:
            return 0.0
        score = self._sparse_vector_to_classifier_score(sparse_vector)
        if score is None:
            return 0.0
        return score

    def _get_drbert_idr_score(self, sequence: str) -> float:
        return self.drbert_scorer.get_drbert_idr_score(sequence)

    
    def _sequence_to_esm2_residue_vector(self, seq: str) -> Optional[torch.Tensor]:
        """
        动态适配任意长度序列的ESM2特征提取
        
        关键改进：
        1. 使用实际序列长度（而非固定Config.PROTEIN_LENGTH）
        2. ESM2会自动处理掩码符号'_'，无需特殊补0逻辑
        """
        # 1. 序列预处理：保留ESM2支持的字符+掩码符号
        allowed_chars = set(self.esm2_alphabet.all_toks) | {"_"}
        seq_processed = ''.join([c if c in allowed_chars else 'X' for c in seq])
        
        # 2. 转换为ESM2模型输入
        batch_data = [("guided_seq", seq_processed)]
        batch_labels, batch_strs, batch_tokens = self.esm2_batch_converter(batch_data)
        batch_tokens = batch_tokens.to(self.device)
        
        try:
            # 3. ESM2前向传播
            with torch.no_grad():
                results = self.esm2_model(batch_tokens, repr_layers=[33])
            
            # 4. 提取残基级表示（去掉特殊token [CLS]和[SEP]）
            token_reps = results["representations"][33]  # (1, L+2, 1280)
            tokens_len = (batch_tokens != self.esm2_alphabet.padding_idx).sum(1).item()
            residue_reps = token_reps[0, 1:tokens_len-1, :].cpu()  # (L, 1280)
            
            # ========== 关键修改：使用实际序列长度 ==========
            actual_length = len(seq)  # 动态获取（如150）
            
            # 5. 处理长度不匹配情况
            if residue_reps.shape[0] < actual_length:
                # ESM2输出短于实际序列（理论上不应该发生，但做容错处理）
                pad_length = actual_length - residue_reps.shape[0]
                print(f"⚠️  ESM2输出长度({residue_reps.shape[0]})短于序列长度({actual_length})，补0")
                residue_reps = torch.cat([
                    residue_reps,
                    torch.zeros(pad_length, residue_reps.shape[1])
                ], dim=0)
            elif residue_reps.shape[0] > actual_length:
                # ESM2输出长于实际序列（截断多余部分）
                print(f"⚠️  ESM2输出长度({residue_reps.shape[0]})超过序列长度({actual_length})，截断")
                residue_reps = residue_reps[:actual_length]
            
            # 6. 返回精确匹配序列长度的特征
            return residue_reps  # (actual_length, 1280)
            
        except Exception as e:
            print(f"⚠️  ESM2编码失败：{str(e)}")
            return None
    
    def _esm2_to_sparse_vector(self, residue_vector: torch.Tensor) -> Optional[torch.Tensor]:
        """
        ESM2残基向量 → SAE稀疏向量
        
        关键改进：
        1. 移除固定长度限制（仅校验特征维度）
        2. 支持任意序列长度（如150、100等）
        """
        self.sae_model.eval()
        try:
            with torch.no_grad():
                # 分批处理（避免显存溢出）
                activations = []
                batch_size = 64
                for i in range(0, residue_vector.shape[0], batch_size):
                    batch = residue_vector[i:i+batch_size].to(self.device)
                    batch_act = self.sae_model.encode(batch)
                    activations.append(batch_act.cpu())
                sparse_vector = torch.cat(activations, dim=0)
            
            # ========== 关键修改：仅校验特征维度，不限制序列长度 ==========
            # 原代码：if sparse_vector.shape != (Config.PROTEIN_LENGTH, Config.FEATURE_DIM)
            # 新代码：只检查第二维（特征维度）
            if sparse_vector.shape[1] != Config.FEATURE_DIM:
                print(f"⚠️  SAE特征维度不匹配：期望{Config.FEATURE_DIM}，实际{sparse_vector.shape[1]}")
                return None
            
            # 允许任意序列长度（如 (150, 40960) 或 (100, 40960)）
            return sparse_vector
            
        except Exception as e:
            print(f"⚠️  SAE编码失败：{str(e)}")
            return None
    
    def _sparse_vector_to_classifier_score(self, sparse_vector: torch.Tensor) -> Optional[float]:
        """稀疏向量 → 分类器AUC加权分数（0~1）
        关键修改：1. 兼容Tensor/float输出 2. 原生Python裁剪数值范围
        """
        final_score = 0.0
        total_auc = 0.0
        sparse_vector = sparse_vector.to(self.device)
        
        # 1. 特征掩码 + 提取3000维重要特征（原逻辑）
        feature_mask = self.binary_mask.bool()  # 3000个1，其余0
        masked_feature = sparse_vector[:, feature_mask]  # (256, 3000) → 仅保留重要特征
        
        # 2. 核心新增：重构40960维完整特征矩阵（重要特征填值，其余补0）
        batch_size = masked_feature.shape[0]  # 256
        full_feature = torch.zeros((batch_size, Config.FEATURE_DIM), dtype=torch.float32).to(self.device)  # (256, 40960)
        full_feature[:, feature_mask] = masked_feature  # 仅3000个重要特征位置填充真实值，其余为0
        
        # 3. 平均池化（残基级→序列级）：从40960维特征池化
        seq_feature = full_feature.mean(dim=0).unsqueeze(0)  # (1, 40960) → 满足模型输入维度要求
        
        # 4. 遍历分类器打分（兼容双类型输出）
        for compare_class in range(14):
            if compare_class == self.target_class:
                continue
            
            model_key = f"class_{self.target_class}_vs_{compare_class}"
            if model_key not in self.classifiers:
                continue
            
            auc_key = f"{self.target_class}_vs_{compare_class}"
            if auc_key not in self.train_auc:
                print(f"⚠️  分类器{model_key}缺失AUC值，跳过")
                continue
            
            # 分类器推理（输入为40960维，匹配模型要求）
            model = self.classifiers[model_key]
            model.to(self.device)
            try:
                with torch.no_grad():
                    pred = model(seq_feature)  # 输入(1,40960)
                    
                    # ========== 核心修复：统一转为float + 原生裁剪 ==========
                    # 步骤1：判断类型，统一转为Python float
                    if isinstance(pred, torch.Tensor):
                        # 处理Tensor：展平→取第一个值→转float
                        pred_flat = pred.flatten()
                        pred_val = pred_flat[0].item() if len(pred_flat) > 0 else 0.0
                    else:
                        # 处理原生float/int：直接赋值
                        pred_val = float(pred)
                    
                    # 步骤2：用Python原生min/max裁剪到[0,1]（替代torch.clip）
                    prob = max(0.0, min(pred_val, 1.0))
                    
            except Exception as e:
                print(f"⚠️  分类器{model_key}推理失败：{str(e)}")
                continue
            
            # AUC加权累加
            auc = self.train_auc[auc_key]
            final_score += prob * auc
            total_auc += auc
        
        if total_auc <= 0:
            print("⚠️  无有效分类器参与打分，返回0分")
            return 0.0
        
        # ========== 额外修复：判断final_score类型（避免float无.item()） ==========
        final_score_val = final_score / total_auc
        # 统一转为float（兼容Tensor/float）
        if isinstance(final_score_val, torch.Tensor):
            return final_score_val.item()
        else:
            return float(final_score_val)

# ================================ 模型加载模块 ================================
class ESM3ModelLoader:
    """ESM3 模型加载器（统一设备管理）"""
    
    def __init__(self, model_name: str = Config.MODEL_NAME):
        self.model_name = model_name
        self.model = None
        self.device = None
        
    def load(self) -> ESM3:
        """加载 ESM3 模型到配置指定设备"""
        try:
            # 设置环境变量
            os.environ["INFRA_PROVIDER"] = Config.INFRA_PROVIDER
            
            # 固定使用配置中的设备（避免与其他模型设备不一致）
            self.device = Config.DEVICE
            print(f"🔧 使用配置设备: {self.device}")
            
            # 加载预训练模型
            print(f"⏳ 正在加载模型 {self.model_name}...")
            self.model = ESM3.from_pretrained(
                model_name=self.model_name,
                device=self.device,
            )
            
            print(f"✅ ESM3模型在 {self.device} 上加载成功！")
            return self.model
            
        except Exception as e:
            print(f"❌ ESM3模型加载失败: {str(e)}")
            raise

# ================================ 蛋白质生成模块 ================================
class ProteinGenerator:
    """蛋白质生成器（仅保留序列生成，删除结构相关逻辑）"""
    
    def __init__(self, model: ESM3):
        self.model = model
        self.device = Config.DEVICE
        
    def create_masked_protein(self, length: int = Config.PROTEIN_LENGTH) -> ESMProtein:
        """创建全掩码肽段（引导生成起始点）"""
        sequence = "_" * length
        protein = ESMProtein(sequence=sequence)
        # print(f"🧬 创建了长度为 {length} 的全掩码肽段")
        return protein
    
    # ========== 新增方法2：创建部分掩码序列 ==========
    def create_partially_masked_protein(self, sequence: str, mask_start: int, mask_end: int) -> ESMProtein:
        seq_len = len(sequence)
        if not (0 <= mask_start < mask_end <= seq_len):
            raise ValueError(...)
        
        masked_seq = (
            sequence[:mask_start] +
            "_" * (mask_end - mask_start) +
            sequence[mask_end:]
        )
        
        protein = ESMProtein(sequence=masked_seq)
        # 简化为一行打印
        print(f"  🧬 掩码区域: [{mask_start}:{mask_end}] (保留{mask_start}个残基)")
        return protein
        
    def generate_with_guidance(
        self,
        starting_protein: ESMProtein,
        num_steps: int = Config.NUM_DECODING_STEPS,
        num_samples: int = Config.NUM_SAMPLES_PER_STEP,
        idr_weight: Optional[float] = None
    ) -> ESMProtein:
        if idr_weight is None:
            idr_weight = Config.IDR_WEIGHT
        if num_steps is None:
            num_steps = Config.NUM_DECODING_STEPS
        if num_samples is None:
            num_samples = Config.NUM_SAMPLES_PER_STEP

        # print(f"  🎯 引导生成中... idr_weight={idr_weight}")

        guided_decoder = ESM3GuidedDecoding(
            client=self.model,
            scoring_function=CustomClassifierScoringFunction(idr_weight=idr_weight)
        )

        generated_protein = guided_decoder.guided_generate(
            protein=starting_protein,
            num_decoding_steps=num_steps,
            num_samples_per_step=num_samples,
            # denoised_prediction_temperature=Config.DENOISED_PREDICTION_TEMP,
            track="sequence",
            verbose=False  # 关闭详细日志
        )

        # 保证生成序列只含标准氨基酸
        filtered_seq = filter_amino_acid_sequence(generated_protein.sequence)
        generated_protein.sequence = filtered_seq

        return generated_protein
        
    def generate_without_guidance(self, starting_protein: ESMProtein, num_steps: int = Config.NUM_DECODING_STEPS) -> ESMProtein:
        # print(f"  🔄 无引导生成中...")
        
        config = GenerationConfig(
            track="sequence",
            num_steps=num_steps,
            # temperature=Config.UNGUIDED_TEMP
        )
        
        generated_protein = self.model.generate(input=starting_protein, config=config)

        filtered_seq = filter_amino_acid_sequence(generated_protein.sequence)
        # 替换为过滤后的序列
        generated_protein.sequence = filtered_seq
        return generated_protein

# ================================ 可视化模块 ================================
class ProteinVisualizer:
    """肽段可视化工具（仅保留序列相关功能）"""
    
    @staticmethod
    def save_sequence(protein: ESMProtein, filename: str):
        """保存肽段序列为FASTA文件（原方法保留）"""
        try:
            with open(filename, "w") as f:
                f.write(f"> {filename}\n")
                f.write(protein.sequence + "\n")
            print(f"💾 序列已保存至：{os.path.abspath(filename)}")
        except Exception as e:
            print(f"❌ 保存FASTA文件失败：{str(e)}")
    
    # ========== 新增方法：保存多条序列到单个FASTA ==========
    @staticmethod
    def save_multiple_sequences(
        sequences: Dict[str, str],  # 键=序列名称, 值=序列
        filename: str
    ):
        """
        保存多条序列到单个FASTA文件
        
        参数:
            sequences: 字典，如 {"full_seq1": "MKTAYIAK...", "full_seq2": "...", ...}
            filename: 输出文件名
        
        示例:
            save_multiple_sequences(
                {
                    "full_seq1": original_seq,
                    "full_seq2": guided_seq,
                    "full_seq3": unguided_seq
                },
                "full_seq123_class_4.fasta"
            )
        """
        try:
            with open(filename, "w") as f:
                for name, seq in sequences.items():
                    f.write(f">{name}\n")
                    # 按80个字符换行（FASTA标准格式）
                    for i in range(0, len(seq), 80):
                        f.write(seq[i:i+80] + "\n")
            
            # print(f"💾 已保存{len(sequences)}条序列至：{os.path.abspath(filename)}")
            # print(f"   序列名称：{', '.join(sequences.keys())}")
        except Exception as e:
            print(f"❌ 保存多序列FASTA文件失败：{str(e)}")

# ================================ 评估模块 ================================
class ProteinEvaluator:
    """肽段评估工具（专注于序列无序性特征）"""
    _scorer_instance = None
    
    @staticmethod
    def analyze_disorder_features(sequence: str, label: str = "肽段"):
        """分析序列的无序性特征（低复杂性、柔性残基占比等）"""
        # 无序肽段常见特征：高比例柔性残基（G、A、S、P等）、低疏水性
        flexible_residues = {'G', 'A', 'S', 'P', 'Q', 'T'}  # 柔性残基
        flex_count = sum(1 for c in sequence if c in flexible_residues)
        flex_ratio = flex_count / len(sequence)
        
        # 低复杂性序列特征：氨基酸种类少（< 10种）
        unique_aa = len(set(sequence))
        
        print(f"\n📊 {label}无序性特征分析:")
        print(f"  柔性残基占比: {flex_ratio:.2%} (G/A/S/P/Q/T)")
        print(f"  独特氨基酸种类: {unique_aa} (无序肽段通常<10)")
        print(f"  序列长度: {len(sequence)}")
        
        # 简单评分：柔性残基占比>60%且独特种类<10 → 高无序性
        is_highly_disordered = flex_ratio > 0.6 and unique_aa < 10
        print(f"  无序性评估: {'高' if is_highly_disordered else '中/低'}")
    
    @staticmethod
    def print_sequence(protein: ESMProtein, label: str = "肽段", max_length: int = 80):
        """打印肽段序列（保留）"""
        seq = protein.sequence
        seq_len = len(seq)
        
        print(f"\n🧬 {label}序列（长度：{seq_len}）:")
        for i in range(0, seq_len, max_length):
            print(f"  {seq[i:i+max_length]}")

    @staticmethod
    def score_final_sequence(sequence: str) -> float:
        seq = filter_amino_acid_sequence(sequence)
        if ProteinEvaluator._scorer_instance is None:
            ProteinEvaluator._scorer_instance = CustomClassifierScoringFunction()
        scorer = ProteinEvaluator._scorer_instance
        try:
            residue_vector = scorer._sequence_to_esm2_residue_vector(seq)
            if residue_vector is None:
                return 0.0
            sparse_vector = scorer._esm2_to_sparse_vector(residue_vector)
            if sparse_vector is None:
                return 0.0
            score = scorer._sparse_vector_to_classifier_score(sparse_vector)
            return score if score is not None else 0.0
        except Exception as e:
            print(f"⚠️  最终序列评分失败：{e}")
            return 0.0

# ================================ 主流程模块 ================================
class ProteinDesignPipeline:
    """无序肽段设计流程（仅支持两种生成模式）"""
    
    def __init__(self):
        self.model_loader = ESM3ModelLoader()
        self.model = None
        self.generator = None
    
    # ========== 方法1：批量从头生成 ==========
    def run_batch_full_mask_generation(
        self,
        num_sequences: int,
        protein_length: int,
        output_dir: str = "batch_full_mask_results"
    ):
        """
        批量从头生成全掩码序列
        
        参数:
            num_sequences: 要生成的序列数量
            protein_length: 每条序列的长度
            output_dir: 输出目录
        """
        print("\n" + "="*80)
        print(f"🚀 方法1：批量从头生成 {num_sequences} 条长度为 {protein_length} 的序列")
        print("="*80)
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        print(f"📁 输出目录：{os.path.abspath(output_dir)}")
        
        # 加载ESM3模型（一次性加载）
        if self.model is None:
            self.model = self.model_loader.load()
            self.generator = ProteinGenerator(self.model)
        
        # 批量生成序列
        all_sequences_dict = {}
        results_summary = []
        
        print(f"\n🔄 开始生成 {num_sequences} 条序列...")
        for i in tqdm(range(num_sequences), desc="生成进度"):
            try:
                # 创建全掩码起始序列
                starting_protein = self.generator.create_masked_protein(protein_length)
                
                # 同时生成引导和无引导序列
                guided_protein = self.generator.generate_with_guidance(starting_protein)
                unguided_protein = self.generator.generate_without_guidance(starting_protein)
                
                # 评分
                score_guided = combined_sequence_score(guided_protein.sequence)
                score_unguided = combined_sequence_score(unguided_protein.sequence)

                # 保存到字典
                seq_id = f"seq_{i+1:04d}"
                all_sequences_dict[f"{seq_id}_guided"] = guided_protein.sequence
                all_sequences_dict[f"{seq_id}_unguided"] = unguided_protein.sequence
                
                # 记录摘要
                results_summary.append({
                    'Sequence_ID': seq_id,
                    'Length': protein_length,
                    'Score_Guided': score_guided,
                    'Score_Unguided': score_unguided,
                    'Score_Diff': score_guided - score_unguided
                })
                
                # 每100条打印一次进度
                if (i + 1) % 100 == 0:
                    avg_guided = np.mean([r['Score_Guided'] for r in results_summary[-100:]])
                    avg_unguided = np.mean([r['Score_Unguided'] for r in results_summary[-100:]])
                    print(f"  ✓ 已生成 {i+1}/{num_sequences} 条 | 近100条平均得分: 引导={avg_guided:.4f}, 无引导={avg_unguided:.4f}")
                
            except Exception as e:
                print(f"  ⚠️  生成第 {i+1} 条序列时出错：{str(e)}")
                continue
        
        # 保存FASTA文件
        self._save_results(all_sequences_dict, results_summary, output_dir, "full_mask")
    
    # ========== 方法2：从CSV批量掩码生成 ==========
    def run_batch_from_csv(
        self,
        csv_file: str,
        output_dir: str = "batch_masked_results",
        mask_ratio: float = 0.9
    ):
        """
        从CSV文件批量掩码生成
        
        参数:
            csv_file: 输入CSV文件路径（需包含 'Genes' 和 'Sequence' 列）
            output_dir: 输出目录
            mask_ratio: 掩码比例（默认0.9，即掩盖90%）
        """
        print("\n" + "="*80)
        print("🚀 方法2：从CSV文件批量掩码生成")
        print("="*80)
        
        # 读取CSV文件
        try:
            df = pd.read_csv(csv_file)
            print(f"✅ 成功读取CSV文件：{csv_file}")
            print(f"   共 {len(df)} 条序列")
        except Exception as e:
            print(f"❌ 读取CSV文件失败：{str(e)}")
            return
        
        # 验证必需列
        if 'Genes' not in df.columns or 'Sequence' not in df.columns:
            print(f"❌ CSV文件缺少必需列：'Genes' 和 'Sequence'")
            return
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        print(f"📁 输出目录：{os.path.abspath(output_dir)}")
        
        # 加载ESM3模型
        if self.model is None:
            self.model = self.model_loader.load()
            self.generator = ProteinGenerator(self.model)
        
        # 定义随机种子起点和步长
        base_seed = 42
        seed_step = 5
        num_seeds = 10
        
        all_sequences_dict = {}
        results_summary = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="批量生成进度"):
            gene_name = row['Genes']
            sequence = row['Sequence']
            seq_len = len(sequence)
            
            print(f"\n{'='*60}")
            print(f"🧬 [{idx+1}/{len(df)}] {gene_name} (长度={seq_len}, 掩盖{mask_ratio*100:.0f}%)")
            print(f"{'='*60}")
            
            try:
                mask_length = int(seq_len * mask_ratio)
                if mask_length <= 0:
                    print(f"⚠️  跳过{gene_name}：序列过短")
                    continue
                
                # 对每条序列，用10个不同种子依次生成
                for i in range(num_seeds):
                    seed = base_seed + i*seed_step
                    random.seed(seed)
                    
                    mask_start = random.randint(0, seq_len - mask_length)
                    mask_end = mask_start + mask_length
                    
                    starting_protein = self.generator.create_partially_masked_protein(
                        sequence, mask_start, mask_end
                    )
                    guided_protein = self.generator.generate_with_guidance(starting_protein)
                    unguided_protein = self.generator.generate_without_guidance(starting_protein)
                    
                    score_original = combined_sequence_score(sequence)
                    score_guided = combined_sequence_score(guided_protein.sequence)
                    score_unguided = combined_sequence_score(unguided_protein.sequence)
                    
                    print(f"  📈 种子 {seed} 得分: 原始={score_original:.3f} | 引导={score_guided:.3f} | 无引导={score_unguided:.3f}")
                    
                    # 保存时区分种子
                    all_sequences_dict[f"{gene_name}_original"] = sequence
                    all_sequences_dict[f"{gene_name}_guided_seed{seed}"] = guided_protein.sequence
                    all_sequences_dict[f"{gene_name}_unguided_seed{seed}"] = unguided_protein.sequence
                    
                    results_summary.append({
                        'Gene': gene_name,
                        'Original_Length': seq_len,
                        'Mask_Start': mask_start,
                        'Mask_End': mask_end,
                        'Seed': seed,
                        'Score_Original': score_original,
                        'Score_Guided': score_guided,
                        'Score_Unguided': score_unguided,
                        'Guided_Improvement': ((score_guided - score_original) / score_original * 100) if score_original > 0 else 0
                    })
                
            except Exception as e:
                print(f"❌ 处理{gene_name}时出错：{str(e)}")
                continue
        
        self._save_results(all_sequences_dict, results_summary, output_dir, "masked")
    def run_batch_prompt_mask_generation(
        self,
        csv_file: str,
        output_dir: str = "batch_prompt_masked_results",
        mask_length: int = 100,
    ):
        """
        基于给定CSV的Prompt，指定区间内用固定种子生成多次随机掩码子序列生成。

        参数:
            csv_file: 包含 Uniprot, Genes, Sequence, Start, End 五列的CSV路径
            output_dir: 输出目录
            mask_length: 掩码长度，默认为100
        """
        print("\n" + "=" * 80)
        print("🚀 方法3：基于Prompt随机掩码区间多次生成")
        print("=" * 80)

        try:
            df = pd.read_csv(csv_file)
            print(f"✅ 成功读取CSV文件：{csv_file}")
            print(f"   共 {len(df)} 条序列")
        except Exception as e:
            print(f"❌ 读取CSV文件失败：{str(e)}")
            return

        required_cols = ["Uniprot", "Genes", "Sequence", "Start", "End"]
        if not all(col in df.columns for col in required_cols):
            print(f"❌ CSV文件缺少必需列: {required_cols}")
            return

        os.makedirs(output_dir, exist_ok=True)
        print(f"📁 输出目录：{os.path.abspath(output_dir)}")

        if self.model is None:
            self.model = self.model_loader.load()
            self.generator = ProteinGenerator(self.model)

        all_sequences_dict = {}
        results_summary = []

        # 设定固定种子参数
        base_seed = 42
        seed_step = 5
        num_seeds = 5

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="批量生成进度"):
            gene_name = row["Genes"]
            sequence = row["Sequence"]
            seq_len = len(sequence)

            start_pos = int(row["Start"]) - 1  # 转为0-based索引
            end_pos = int(row["End"])

            if not (0 <= start_pos < end_pos <= seq_len):
                print(f"⚠️ {gene_name} 的 Start/End 不合法，跳过")
                continue

            interval_len = end_pos - start_pos
            if interval_len < mask_length:
                print(f"⚠️ {gene_name} 在[{start_pos+1}, {end_pos}]范围内长度不足{mask_length}，跳过")
                continue

            print(f"\n{'=' * 60}")
            print(f"🧬 [{idx + 1}/{len(df)}] {gene_name} (Prompt长度={seq_len}, 掩码区间长度={mask_length})")
            print(f"{'=' * 60}")

            try:
                for i in range(num_seeds):
                    seed = base_seed + i * seed_step
                    random.seed(seed)

                    mask_start_in_interval = random.randint(0, interval_len - mask_length)
                    mask_start = start_pos + mask_start_in_interval
                    mask_end = mask_start + mask_length

                    masked_seq = (
                        sequence[:mask_start] + "_" * mask_length + sequence[mask_end:]
                    )

                    starting_protein = self.generator.create_partially_masked_protein(
                        masked_seq, mask_start, mask_end
                    )
                    guided_protein = self.generator.generate_with_guidance(starting_protein)
                    unguided_protein = self.generator.generate_without_guidance(starting_protein)

                    score_original = combined_sequence_score(sequence)
                    score_guided = combined_sequence_score(guided_protein.sequence)
                    score_unguided = combined_sequence_score(unguided_protein.sequence)

                    print(
                        f"  📈 种子 {seed} 得分: 原始={score_original:.3f} | 引导={score_guided:.3f} | 无引导={score_unguided:.3f}"
                    )

                    key_base = (
                        f"{gene_name}_prompt_mask_{mask_start + 1}_{mask_end}_seed{seed}"
                    )
                    all_sequences_dict[f"{key_base}_original"] = sequence
                    all_sequences_dict[f"{key_base}_guided"] = guided_protein.sequence
                    all_sequences_dict[f"{key_base}_unguided"] = unguided_protein.sequence

                    results_summary.append(
                        {
                            "Gene": gene_name,
                            "Original_Length": seq_len,
                            "Prompt_Start": start_pos + 1,
                            "Prompt_End": end_pos,
                            "Mask_Start": mask_start + 1,
                            "Mask_End": mask_end,
                            "Mask_Length": mask_length,
                            "Seed": seed,
                            "Score_Original": score_original,
                            "Score_Guided": score_guided,
                            "Score_Unguided": score_unguided,
                            "Guided_Improvement": (
                                (score_guided - score_original) / score_original * 100
                                if score_original > 0
                                else 0
                            ),
                        }
                    )

            except Exception as e:
                print(f"❌ 处理{gene_name}时出错：{str(e)}")
                continue

        self._save_results(all_sequences_dict, results_summary, output_dir, "prompt_masked")
    
    # ========== 通用保存方法（复用代码） ==========
    def _save_results(self, sequences_dict, results_summary, output_dir, tag):
        """
        保存FASTA和CSV结果（两种方法复用）
        
        参数:
            sequences_dict: 序列字典
            results_summary: 结果摘要列表
            output_dir: 输出目录
            tag: 文件标签（full_mask 或 masked）
        """
        if not sequences_dict:
            print("\n⚠️  没有成功生成任何序列")
            return
        
        # 保存FASTA文件
        output_fasta = os.path.join(
            output_dir,
            f"all_sequences_{tag}_class_{Config.TARGET_CLASS}.fasta"
        )
        ProteinVisualizer.save_multiple_sequences(sequences_dict, output_fasta)
        print(f"\n💾 所有序列已保存至：{output_fasta}")
        
        # 保存摘要CSV
        if results_summary:
            summary_df = pd.DataFrame(results_summary)
            summary_file = os.path.join(
                output_dir,
                f"generation_summary_{tag}_class_{Config.TARGET_CLASS}.csv"
            )
            summary_df.to_csv(summary_file, index=False)
            
            print("\n" + "="*80)
            print("📊 生成完成！统计信息：")
            print("="*80)
            print(f"  成功生成序列数: {len(results_summary)}")
            
            # 根据tag显示不同的统计信息
            if tag == "full_mask":
                print(f"  引导平均得分: {summary_df['Score_Guided'].mean():.4f}")
                print(f"  无引导平均得分: {summary_df['Score_Unguided'].mean():.4f}")
            else:
                print(f"  原始平均得分: {summary_df['Score_Original'].mean():.4f}")
                print(f"  引导平均得分: {summary_df['Score_Guided'].mean():.4f}")
                print(f"  平均改进: {summary_df['Guided_Improvement'].mean():.2f}%")
            
            print(f"\n💾 结果摘要已保存至：{summary_file}")


# ================================ 主程序入口 ================================
def main():
    """主函数：两种生成方法"""
    torch.manual_seed(42)
    np.random.seed(42)
    
    pipeline = ProteinDesignPipeline()
    
    # ========== 方法1：批量从头生成 ==========
    pipeline.run_batch_full_mask_generation(
        num_sequences=500,          # 生成1000条序列
        protein_length=70,          # 每条长度100
        output_dir="batch_full_mask_results_70aa_20260316"
    )
    
    # ========== 方法2：从CSV批量掩码生成 ==========
    # pipeline.run_batch_from_csv(
    #     csv_file="activated_idr_regions_denovo.csv",  # 输入CSV文件
    #     output_dir="batch_masked_results_060",
    #     mask_ratio=0.6  # 掩盖90%
    # )


    # ========== 方法3：基于prompt生成 ==========
    # pipeline.run_batch_prompt_mask_generation(
    #     csv_file="activated_idr_regions_prompt.csv",
    #     output_dir="batch_prompt_masked_results_mask50",
    #     mask_length=50
    # )

if __name__ == "__main__":
    main()
