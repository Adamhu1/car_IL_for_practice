import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

class BCTrainer:
    def __init__(self, model, model_type='state'):
        self.model = model
        self.model_type = model_type
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.1)
        
        self.train_losses = []
        self.val_accuracies = []
        
    def prepare_data(self, dataset, train_ratio=0.8):
        """准备训练和验证数据"""
        states = np.array(dataset['states'])
        actions = np.array(dataset['actions'])
        pixel_states = np.array(dataset['pixel_states'])
        
        # 分割训练集和验证集
        n_total = len(actions)
        n_train = int(n_total * train_ratio)
        indices = np.random.permutation(n_total)
        
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]
        
        if self.model_type == 'state':
            # 使用状态向量
            X_train = torch.FloatTensor(states[train_indices])
            X_val = torch.FloatTensor(states[val_indices])
        elif self.model_type == 'pixel':
            # 使用像素状态
            X_train = torch.FloatTensor(pixel_states[train_indices]).unsqueeze(1)
            X_val = torch.FloatTensor(pixel_states[val_indices]).unsqueeze(1)
        elif self.model_type == 'ensemble':
            # 使用两者
            X_train = (torch.FloatTensor(states[train_indices]), 
                      torch.FloatTensor(pixel_states[train_indices]).unsqueeze(1))
            X_val = (torch.FloatTensor(states[val_indices]), 
                    torch.FloatTensor(pixel_states[val_indices]).unsqueeze(1))
        
        y_train = torch.LongTensor(actions[train_indices])
        y_val = torch.LongTensor(actions[val_indices])
        
        # 创建数据加载器
        if self.model_type == 'ensemble':
            train_dataset = TensorDataset(X_train[0], X_train[1], y_train)
            val_dataset = TensorDataset(X_val[0], X_val[1], y_val)
        else:
            train_dataset = TensorDataset(X_train, y_train)
            val_dataset = TensorDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        print(f"训练集大小: {len(train_dataset)}")
        print(f"验证集大小: {len(val_dataset)}")
        
        return train_loader, val_loader
    
    def train(self, train_loader, val_loader, epochs=100):
        """训练模型"""
        print("开始训练...")
        
        for epoch in range(epochs):
            # 训练阶段
            self.model.train()
            train_loss = 0.0
            
            for batch in train_loader:
                if self.model_type == 'ensemble':
                    states, pixels, actions = batch
                    states, pixels, actions = (states.to(self.device), 
                                             pixels.to(self.device), 
                                             actions.to(self.device))
                    self.optimizer.zero_grad()
                    outputs = self.model(states, pixels)
                else:
                    inputs, actions = batch
                    inputs, actions = inputs.to(self.device), actions.to(self.device)
                    self.optimizer.zero_grad()
                    outputs = self.model(inputs)
                
                loss = self.criterion(outputs, actions)
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
            
            # 验证阶段
            self.model.eval()
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    if self.model_type == 'ensemble':
                        states, pixels, actions = batch
                        states, pixels, actions = (states.to(self.device), 
                                                 pixels.to(self.device), 
                                                 actions.to(self.device))
                        outputs = self.model(states, pixels)
                    else:
                        inputs, actions = batch
                        inputs, actions = inputs.to(self.device), actions.to(self.device)
                        outputs = self.model(inputs)
                    
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += actions.size(0)
                    val_correct += (predicted == actions).sum().item()
            
            avg_train_loss = train_loss / len(train_loader)
            val_accuracy = 100 * val_correct / val_total
            
            self.train_losses.append(avg_train_loss)
            self.val_accuracies.append(val_accuracy)
            
            if epoch % 10 == 0:
                print(f'Epoch [{epoch}/{epochs}], '
                      f'Loss: {avg_train_loss:.4f}, '
                      f'Val Acc: {val_accuracy:.2f}%')
            
            self.scheduler.step()
        
        print("训练完成!")
    
    def save_model(self, filename=None):
        """保存模型"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"bc_model_{self.model_type}_{timestamp}.pth"
        
        os.makedirs('models', exist_ok=True)
        filepath = os.path.join('models', filename)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_type': self.model_type,
            'train_losses': self.train_losses,
            'val_accuracies': self.val_accuracies
        }, filepath)
        
        print(f"模型已保存到: {filepath}")
        return filepath
    
    def load_model(self, filename):
        """加载模型"""
        filepath = os.path.join('models', filename)
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_accuracies = checkpoint.get('val_accuracies', [])
        print(f"模型已加载: {filepath}")
    
    def plot_training_history(self):
        """绘制训练历史"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # 损失曲线
        ax1.plot(self.train_losses)
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.grid(True)
        
        # 准确率曲线
        ax2.plot(self.val_accuracies)
        ax2.set_title('Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()