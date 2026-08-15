import torch.nn as nn
import torch.optim as optim
import torch


class MLPModel(nn.Module):
    def __init__(self, input_size=1280, hidden_size1=512, hidden_size2=512, output_size=1):
        """
        初始化 MLP 模型。

        参数:
            input_size (int): 输入特征维度，默认为 1280。
            hidden_size1 (int): 第一隐藏层的维度，默认为 512。
            hidden_size2 (int): 第二隐藏层的维度，默认为 256。
            output_size (int): 输出维度，默认为 1。
        """
        super(MLPModel, self).__init__()

        # 第一层全连接，从 input_size 到 hidden_size1
        self.fc1 = nn.Linear(input_size, hidden_size1)

        # 第二层全连接，从 hidden_size1 到 hidden_size2
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)

        # 输出层，从 hidden_size2 到 output_size
        self.fc3 = nn.Linear(hidden_size2, output_size)

        # 激活函数
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        # Dropout
        self.dropout = nn.Dropout(0.5)

        # 批归一化
        self.bn1 = nn.LayerNorm(hidden_size1)
        self.bn2 = nn.LayerNorm(hidden_size2)

    def forward(self, x):
        """
        前向传播。

        参数:
            x (torch.Tensor): 输入张量，形状为 (batch_size, input_size)。

        返回:
            torch.Tensor: 输出张量，形状为 (batch_size, output_size)。
        """
        # 第一层
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)

        # 第二层
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)

        # 输出层
        x = self.fc3(x)
        x = self.sigmoid(x)

        return x  # 去掉多余的维度


class MLPWithL1FeatureSelection(nn.Module):
    def __init__(self, input_size=1280, hidden_size1=512, hidden_size2=512, output_size=1):
        super().__init__()
        # 原MLP结构保持不变
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, output_size)
        
        # 新增特征选择层
        self.feature_selector = nn.Linear(input_size, 1, bias=False)
        
        # 原激活和归一化层
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.5)
        self.bn1 = nn.LayerNorm(hidden_size1)
        self.bn2 = nn.LayerNorm(hidden_size2)
    
    def forward(self, x):
        # 特征重要性加权
        feature_scores = torch.abs(self.feature_selector(x))
        x_weighted = x * feature_scores
        
        # 原MLP前向传播
        x = self.fc1(x_weighted)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x
    
    def get_feature_importance(self):
        # 返回特征重要性
        return torch.abs(self.feature_selector.weight).squeeze()
    

    def reset_final_layer(self):
        """重置最后分类层（用于增量学习）"""
        nn.init.xavier_normal_(self.fc3.weight)
        if self.fc3.bias is not None:
            nn.init.zeros_(self.fc3.bias)



class MLPWithL1FeatureSelection_for_mask(nn.Module):
    def __init__(self, input_size=1280, hidden_size1=512, hidden_size2=512, output_size=1):
        super().__init__()
        # 保持原有结构
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, output_size)
        
        # 特征选择层
        self.feature_selector = nn.Linear(input_size, 1, bias=False)
        
        # 激活和归一化层
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.5)
        self.bn1 = nn.LayerNorm(hidden_size1)
        self.bn2 = nn.LayerNorm(hidden_size2)
        
        # 新增：遮蔽适配参数
        self.mask_adaptation = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
    
    def forward(self, x, mask_ratio=None):
        """
        支持遮蔽比例的前向传播
        
        Args:
            x (torch.Tensor): 输入特征
            mask_ratio (float, optional): 遮蔽比例 
        """
        # 特征重要性加权
        feature_scores = torch.abs(self.feature_selector(x))
        x_weighted = x * feature_scores
        
        # 遮蔽适配
        if mask_ratio is not None and mask_ratio > 0:
            x_weighted = self.mask_adaptation(x_weighted)
        
        # 原MLP前向传播
        x = self.fc1(x_weighted)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x
    
    def get_feature_importance(self):
        """返回特征重要性"""
        return torch.abs(self.feature_selector.weight).squeeze()
    
    def reset_final_layer(self):
        """重置最后分类层（用于增量学习）"""
        nn.init.xavier_normal_(self.fc3.weight)
        if self.fc3.bias is not None:
            nn.init.zeros_(self.fc3.bias)
    
    def create_mask(self, x, mask_ratio):
        """
        创建遮蔽掩码
        
        Args:
            x (torch.Tensor): 输入特征
            mask_ratio (float): 遮蔽比例
        
        Returns:
            torch.Tensor: 遮蔽后的特征
        """
        # 生成随机遮蔽掩码
        mask = torch.rand(x.shape) < mask_ratio
        
        # 在遮蔽区域使用零填充
        x_masked = x.clone()
        x_masked[mask] = 0
        
        return x_masked, mask
class AttentionMLPModel(nn.Module):
    def __init__(
        self, 
        input_size=1280, 
        hidden_sizes=[512, 256, 128],
        attention_size=64,
        dropout_rate=0.5
    ):
        super(AttentionMLPModel, self).__init__()
        
        # 特征提取层
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.BatchNorm1d(hidden_sizes[0]),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        
        # 自注意力机制
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_sizes[0], 
            num_heads=4,
            dropout=dropout_rate,
            batch_first=True
        )
        
        # 后续全连接层
        layers = []
        prev_size = hidden_sizes[0]
        for hidden_size in hidden_sizes[1:]:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.GELU(),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size
        
        self.classifier = nn.Sequential(*layers)
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(prev_size, 1),
            nn.Sigmoid()
        )
        
        # 权重初始化
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        # 特征提取
        features = self.feature_extractor(x)
        
        # 自注意力
        attended_features, _ = self.attention(
            features.unsqueeze(1), 
            features.unsqueeze(1), 
            features.unsqueeze(1)
        )
        attended_features = attended_features.squeeze(1)
        
        # 分类
        classified_features = self.classifier(attended_features)
        
        # 输出
        return self.output_layer(classified_features).squeeze()
    

class MultilabelMLPModel(nn.Module):
    def __init__(self, input_size=1280, hidden_size1=512, hidden_size2=256, num_classes=18):
        """
        初始化 MLP 模型（多标签分类任务）。
        参数:
            input_size (int): 输入特征维度，默认为 1280。
            hidden_size1 (int): 第一隐藏层的维度，默认为 512。
            hidden_size2 (int): 第二隐含层的维度，默认为 256。
            num_classes (int): 类别数，默认为 18。
        """
        super(MultilabelMLPModel, self).__init__()
        # 第一层全连接，从 input_size 到 hidden_size1
        self.fc1 = nn.Linear(input_size, hidden_size1)
        # 第二层全连接，从 hidden_size1 到 hidden_size2
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        # 输出层，从 hidden_size2 到 num_classes
        self.fc3 = nn.Linear(hidden_size2, num_classes)
        # 激活函数
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()  # 多标签激活函数
        # Dropout
        self.dropout = nn.Dropout(0.5)
        # 批归一化
        self.bn1 = nn.BatchNorm1d(hidden_size1)
        self.bn2 = nn.BatchNorm1d(hidden_size2)

    def forward(self, x):
        """
        前向传播。
        参数:
            x (torch.Tensor): 输入张量，形状为 (batch_size, input_size)。
        返回:
            torch.Tensor: 输出张量，形状为 (batch_size, num_classes)。
        """
        # 第一层
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        # 第二层
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        # 输出层
        x = self.fc3(x)
        x = self.sigmoid(x)  # 输出每个标签的概率
        return x