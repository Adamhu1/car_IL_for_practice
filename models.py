import torch
import torch.nn as nn
import torch.nn.functional as F

class BehaviorCloningModel(nn.Module):
    """行为克隆模型 - 状态向量输入"""
    def __init__(self, input_size=7, hidden_size=128, output_size=3):
        super(BehaviorCloningModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            
            nn.Linear(hidden_size//2, output_size)
        )
    
    def forward(self, x):
        return self.network(x)

class CNNBehaviorCloningModel(nn.Module):
    """CNN行为克隆模型 - 图像输入"""
    def __init__(self, input_channels=1, output_size=3):
        super(CNNBehaviorCloningModel, self).__init__()
        self.conv_layers = nn.Sequential(
            # 输入: 84x84
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            # 20x20
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            # 9x9
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            # 7x7
        )
        
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, output_size)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # 展平
        return self.fc_layers(x)

class EnsembleModel(nn.Module):
    """集成模型 - 同时使用状态向量和图像"""
    def __init__(self, state_size=7, image_channels=1, output_size=3):
        super(EnsembleModel, self).__init__()
        
        # 图像分支
        self.image_branch = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        
        # 状态向量分支
        self.state_branch = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        
        # 融合层
        self.fusion_layers = nn.Sequential(
            nn.Linear(64 * 7 * 7 + 64, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, output_size)
        )
    
    def forward(self, state, image):
        # 处理图像
        image_features = self.image_branch(image)
        image_features = image_features.view(image_features.size(0), -1)
        
        # 处理状态向量
        state_features = self.state_branch(state)
        
        # 融合特征
        combined = torch.cat([image_features, state_features], dim=1)
        return self.fusion_layers(combined)