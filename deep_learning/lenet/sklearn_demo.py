"""LeNet（PyTorch 简单版）: 在 digits 数据集上做图像分类。"""


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

    class LeNet(nn.Module):
        """教学常见 LeNet-5 结构。"""

        def __init__(self, num_classes=10):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 6, kernel_size=5),
                nn.ReLU(),
                nn.AvgPool2d(kernel_size=2, stride=2),
                nn.Conv2d(6, 16, kernel_size=5),
                nn.ReLU(),
                nn.AvgPool2d(kernel_size=2, stride=2),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(16 * 5 * 5, 120),
                nn.ReLU(),
                nn.Linear(120, 84),
                nn.ReLU(),
                nn.Linear(84, num_classes),
            )

        def forward(self, x):
            return self.classifier(self.features(x))

    # digits 原图是 8x8，这里放大到 32x32，便于演示 LeNet 标准结构。
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
    y_train = torch.tensor(y_train_np, dtype=torch.long)
    y_test = torch.tensor(y_test_np, dtype=torch.long)

    model = LeNet(num_classes=10)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for _ in range(10):
        logits = model(x_train)
        loss = criterion(logits, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        pred = model(x_test).argmax(dim=1)
        acc = (pred == y_test).float().mean().item()

    print("=== LeNet Demo ===")
    print(f"测试准确率={acc:.4f}")


if __name__ == "__main__":
    main()
