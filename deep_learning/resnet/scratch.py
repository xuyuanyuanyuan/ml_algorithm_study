"""ResNet（手写教学简化版）: 用残差块演示“跨层直连”思想。"""


def main():
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
    except ImportError:
        print("缺少依赖，请安装: pip install torch")
        return

    class ResidualBlockTeaching(nn.Module):
        """教学残差块：输出 = F(x) + x（或投影后的 x）。"""

        def __init__(self, in_channels, out_channels, stride=1):
            super().__init__()
            self.conv1 = nn.Conv2d(
                in_channels, out_channels, kernel_size=3, stride=stride, padding=1
            )
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            self.act = nn.ReLU()
            self.shortcut = None
            if stride != 1 or in_channels != out_channels:
                self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)

        def forward(self, x):
            residual = x if self.shortcut is None else self.shortcut(x)
            out = self.act(self.conv1(x))
            out = self.conv2(out)
            return self.act(out + residual)

    class TeachingResNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = nn.Conv2d(3, 16, kernel_size=3, padding=1)
            self.block1 = ResidualBlockTeaching(16, 16)
            self.block2 = ResidualBlockTeaching(16, 32, stride=2)
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(32, 2)
            self.act = nn.ReLU()

        def forward(self, x):
            x = self.act(self.stem(x))
            x = self.block1(x)
            x = self.block2(x)
            x = self.pool(x).view(x.size(0), -1)
            return self.fc(x)

    # 教学数据：根据通道 0 与通道 1 的均值差生成二分类标签。
    torch.manual_seed(42)
    x = torch.randn(320, 3, 32, 32)
    y = (x[:, 0].mean(dim=(1, 2)) > x[:, 1].mean(dim=(1, 2))).long()

    model = TeachingResNet()
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
    print("=== ResNet Scratch ===")
    print(f"训练准确率={acc:.4f}")


if __name__ == "__main__":
    main()
