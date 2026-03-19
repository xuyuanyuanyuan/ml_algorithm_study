"""VGG（PyTorch 简化版）: 在 digits 数据上演示“堆叠 3x3 卷积”。"""


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

    def vgg_block(in_channels, out_channels):
        # VGG 典型模块：两层 3x3 卷积后接池化。
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

    class TinyVGG(nn.Module):
        def __init__(self, num_classes=10):
            super().__init__()
            self.features = nn.Sequential(
                vgg_block(3, 32),
                vgg_block(32, 64),
                vgg_block(64, 128),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128 * 4 * 4, 256),
                nn.ReLU(),
                nn.Linear(256, num_classes),
            )

        def forward(self, x):
            return self.classifier(self.features(x))

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

    model = TinyVGG(num_classes=10)
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

    print("=== VGG Demo ===")
    print(f"测试准确率={acc:.4f}")


if __name__ == "__main__":
    main()
