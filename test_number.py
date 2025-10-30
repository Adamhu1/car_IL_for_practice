# ========== 1. 导入必要库 ==========
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# ========== 2. 设备设置（优先用GPU） ==========
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'使用设备: {device}')

# ========== 3. 数据预处理与加载 ==========
# 定义图像变换：转为Tensor + 归一化（[0,1] → [-1,1]）
transform = transforms.Compose([
    transforms.ToTensor(),  # 转为Tensor，形状 [C, H, W]，值[0,1]
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST均值与标准差，归一化到[-1,1]
])

# 下载并加载训练集
train_dataset = datasets.MNIST(
    root='./data',           # 数据保存路径
    train=True,              # 加载训练集
    download=True,           # 自动下载
    transform=transform      # 应用变换
)

# 下载并加载测试集
test_dataset = datasets.MNIST(
    root='./data',
    train=False,             # 加载测试集
    download=True,
    transform=transform
)

# 创建DataLoader（批量加载、打乱）
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# =========================================================================
# === 可选：可视化一个数据样本（在数据加载后，模型定义前插入）===========
# =========================================================================

# 从训练集中获取第一个样本：一个图像张量和它的标签
sample_image_tensor, sample_label = train_dataset[0]

# 注意：为了正确显示，我们需要反转之前应用的归一化操作。
# 归一化公式是：output = (input - mean) / std
# 反转归一化公式是：input = output * std + mean
# MNIST 的均值和标准差是：
mean = 0.1307
std = 0.3081

# 反转归一化
# sample_image_tensor 的形状是 [1, 28, 28]
# 我们需要将其转换为 [28, 28] 的 NumPy 数组才能用 matplotlib 显示灰度图
image_for_display = sample_image_tensor * std + mean # 反归一化到 [0, 1] 范围
image_for_display = image_for_display.squeeze(0).numpy() # 移除通道维度并转为 NumPy

# 显示图像
plt.figure(figsize=(3, 3)) # 设置图片大小
plt.imshow(image_for_display, cmap='gray') # 使用灰度色图
plt.title(f'真实标签: {sample_label}') # 设置标题为真实标签
plt.axis('off') # 关闭坐标轴
plt.show() # 显示图片
# =========================================================================


# ========== 4. 定义CNN模型 ==========
class MNIST_CNN(nn.Module):
    def __init__(self):
        super(MNIST_CNN, self).__init__()
        # 第一个卷积块：输入1通道（灰度图），输出32通道
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)  # 输出: 32x28x28
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)                 # 输出: 32x14x14

        # 第二个卷积块：输入32通道，输出64通道
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1) # 输出: 64x14x14
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)                 # 输出: 64x7x7

        # 全连接层：输入64*7*7，输出128
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.relu3 = nn.ReLU()
        # 输出层：10类（数字0~9）
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # 输入 x: [batch, 1, 28, 28]
        x = self.pool1(self.relu1(self.conv1(x)))  # [batch, 32, 14, 14]
        x = self.pool2(self.relu2(self.conv2(x)))  # [batch, 64, 7, 7]
        x = x.view(-1, 64 * 7 * 7)                # 展平 → [batch, 64*7*7]
        x = self.relu3(self.fc1(x))                # [batch, 128]
        x = self.fc2(x)                            # [batch, 10]
        return x

# ========== 5. 初始化模型、损失函数、优化器 ==========
model = MNIST_CNN().to(device)
criterion = nn.CrossEntropyLoss()  # 多分类交叉熵损失
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam优化器

# ========== 6. 训练函数 ==========
def train(epoch):
    model.train()  # 设置为训练模式（启用Dropout/BatchNorm）
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):#相当于for data in data_loader:
        data, target = data.to(device), target.to(device)  # 转移到GPU

        optimizer.zero_grad()             # 梯度清零
        output = model(data)              # 前向传播
        loss = criterion(output, target)  # 计算损失
        loss.backward()                   # 反向传播
        optimizer.step()                  # 更新参数

        total_loss += loss.item()
        pred = output.argmax(dim=1)       # 取概率最大的类别
        correct += pred.eq(target).sum().item()  # 统计正确数
        total += target.size(0)

        if batch_idx % 100 == 0:
            print(f'训练轮次: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\t损失: {loss.item():.6f}')

    accuracy = 100. * correct / total
    avg_loss = total_loss / len(train_loader)
    print(f'====> 训练轮次: {epoch}\t平均损失: {avg_loss:.4f}\t准确率: {accuracy:.2f}%')

# ========== 7. 测试函数 ==========
def test():
    model.eval()  # 设置为评估模式（禁用Dropout/BatchNorm）
    test_loss = 0
    correct = 0

    with torch.no_grad():  # 关闭梯度计算（节省内存+加速）
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # 累加损失
            pred = output.argmax(dim=1)                    # 预测标签
            correct += pred.eq(target).sum().item()        # 统计正确数

    test_loss /= len(test_loader)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'\n测试集\t平均损失: {test_loss:.4f}\t准确率: {accuracy:.2f}%\n')
    return accuracy

# ========== 8. 主训练循环 ==========
if __name__ == "__main__":
    print("开始训练 MNIST CNN 分类器...")
    best_accuracy = 0.0

    for epoch in range(1, 6):  # 训练5轮
        train(epoch)
        accuracy = test()

        # 保存最佳模型
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), 'mnist_cnn_best.pth')
            print(f'模型已保存，最佳准确率: {best_accuracy:.2f}%\n')

    print(f'训练完成！最终最佳测试准确率: {best_accuracy:.2f}%')
