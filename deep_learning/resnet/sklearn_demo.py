"""ResNet（PyTorch 简化版）: 在 digits 数据上演示残差连接。"""


def main():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        import torch.optim as optim
        from sklearn.datasets import load_digits
        from sklearn.model_selection import train_test_split
    except ImportError:
        print("缺少依赖，请安装: pip install torch scikit-learn")
        return

    class BasicBlock(nn.Module):
        """最小残差块：两层卷积 + 跳连。"""

        def __init__(self, in_channels, out_channels, stride=1):
            super().__init__()
            self.conv1 = nn.Conv2d(
                in_channels, out_channels, kernel_size=3, stride=stride, padding=1
            )
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm2d(out_channels)
            self.relu = nn.ReLU()
            self.downsample = None
            if stride != 1 or in_channels != out_channels:
                self.downsample = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                    nn.BatchNorm2d(out_channels),
                )

        def forward(self, x):
            identity = x
            out = self.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            if self.downsample is not None:
                identity = self.downsample(x)
            out = self.relu(out + identity)
            return out

    class TinyResNet(nn.Module):
        def __init__(self, num_classes=10):
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
            )
            self.layer1 = nn.Sequential(BasicBlock(32, 32), BasicBlock(32, 32))
            self.layer2 = nn.Sequential(BasicBlock(32, 64, stride=2), BasicBlock(64, 64))
            self.layer3 = nn.Sequential(BasicBlock(64, 128, stride=2))
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(128, num_classes)

        def forward(self, x):
            x = self.stem(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.pool(x).view(x.size(0), -1)
            return self.fc(x)

    data = load_digits()
    x = data.images.astype("float32") / 16.0
    y = data.target
    x_train_np, x_test_np, y_train_np, y_test_np = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    x_train = torch.tensor(x_train_np, dtype=torch.float32).unsqueeze(1)
    x_test = torch.tensor(x_test_np, dtype=torch.float32).unsqueeze(1)
    x_train = F.interpolate(x_train, size=(32, 32), mode="bilinear", align_corners=False)
    x_test = F.interpolate(x_test, size=(32, 32), mode="bilinear", align_corners=False)
    x_train = x_train.repeat(1, 3, 1, 1)
    x_test = x_test.repeat(1, 3, 1, 1)
    y_train = torch.tensor(y_train_np, dtype=torch.long)
    y_test = torch.tensor(y_test_np, dtype=torch.long)

    model = TinyResNet(num_classes=10)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for _ in range(8):
        logits = model(x_train)
        loss = criterion(logits, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        pred = model(x_test).argmax(dim=1)
        acc = (pred == y_test).float().mean().item()

    print("=== ResNet Demo ===")
    print(f"测试准确率={acc:.4f}")


if __name__ == "__main__":
    main()
