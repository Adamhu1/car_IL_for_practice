import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False  # 使负号正常显示
# 超级简单的模仿学习例子：学习一个正弦波
class SimpleImitation:
    def __init__(self):
        # 创建"专家"数据：y = sin(x)
        self.x = np.linspace(0, 4*np.pi, 1000)
        self.y = np.sin(self.x)
        
        # 添加一些噪声模拟不完美的专家
        noise = np.random.normal(0, 0.1, self.y.shape)
        self.y_expert = self.y + noise
        
    def create_model(self):
        """创建一个简单的神经网络"""
        return nn.Sequential(
            nn.Linear(1, 128),  # 输入1个值(x)
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(), 
            nn.Linear(128, 64),   # 输出1个值(y)
            nn.ReLU(), 
            nn.Linear(64, 1)
        )
    
    def train(self):
        model = self.create_model()
        criterion = nn.MSELoss()  # 回归问题用均方误差
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # 准备数据
        x_tensor = torch.FloatTensor(self.x).unsqueeze(1)
        x_mean, x_std = x_tensor.mean(), x_tensor.std()
        x_norm = (x_tensor - x_mean) / (x_std + 1e-8)
        y_tensor = torch.FloatTensor(self.y_expert).unsqueeze(1)


        dataset = torch.utils.data.TensorDataset(x_norm, y_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
        
        losses = []
        for epoch in range(2000):
            epoch_loss = 0.0
            for xb, yb in loader:
                preds = model(xb)
                loss = criterion(preds, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * xb.size(0)
            epoch_loss /= len(loader.dataset)
            losses.append(epoch_loss)
            if epoch % 200 == 0:
                print(f'Epoch {epoch}, Loss: {epoch_loss:.4f}')
        
        # 画图时对输入做相同归一化
        self._inference_plot(model, x_mean, x_std, losses)
        return model
    
    def _inference_plot(self, model, x_mean, x_std, losses):
        with torch.no_grad():
            x_tensor = torch.FloatTensor(self.x).unsqueeze(1)
            x_norm = (x_tensor - x_mean) / (x_std + 1e-8)
            predictions = model(x_norm).numpy()
        
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(self.x, self.y, 'b-', label='真实函数', linewidth=2)
        plt.plot(self.x, self.y_expert, 'ro', label='专家数据', markersize=1, alpha=0.5)
        plt.plot(self.x, predictions, 'g-', label='模型学习', linewidth=2)
        plt.legend()
        plt.title('模仿学习结果')
        
        plt.subplot(1, 2, 2)
        plt.plot(losses)
        plt.title('训练损失')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        
        plt.tight_layout()
        plt.show()

# 运行这个简单例子
if __name__ == "__main__":
    imitation = SimpleImitation()
    model = imitation.train()