"""LeNet（手写教学简化版）: 用最小代码理解卷积 + 池化 + 全连接。"""


def main():
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
    except ImportError:
        print("缺少依赖，请安装: pip install torch")
        return

    class TeachingLeNet(nn.Module):
        """教学型 LeNet，保留核心结构。"""

        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
            self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
            self.pool = nn.AvgPool2d(2, 2)
            self.act = nn.ReLU()
            self.fc1 = nn.Linear(16 * 5 * 5, 64)
            self.fc2 = nn.Linear(64, 2)

        def forward(self, x):
            x = self.pool(self.act(self.conv1(x)))
            x = self.pool(self.act(self.conv2(x)))
            x = x.view(x.size(0), -1)
            x = self.act(self.fc1(x))
            return self.fc2(x)

    # 生成教学用假数据：根据中心区域均值构造二分类标签。
    torch.manual_seed(42)
    x = torch.randn(240, 1, 32, 32)
    y = (x[:, :, 8:24, 8:24].mean(dim=(1, 2, 3)) > 0).long()

    model = TeachingLeNet()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, 121):
        logits = model(x)
        loss = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 40 == 0:
            print(f"第 {epoch} 轮, loss={loss.item():.6f}")

    acc = (model(x).argmax(dim=1) == y).float().mean().item()
    print("=== LeNet Scratch ===")
    print(f"训练准确率={acc:.4f}")


if __name__ == "__main__":
    main()
