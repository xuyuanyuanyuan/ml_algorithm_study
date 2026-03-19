"""AlexNet（手写教学简化版）: 用小网络理解深层 CNN 的核心结构。"""


def main():
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
    except ImportError:
        print("缺少依赖，请安装: pip install torch")
        return

    class TeachingAlexNet(nn.Module):
        """教学版 AlexNet，突出“多层卷积 + 全连接”。"""

        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(64 * 4 * 4, 64),
                nn.ReLU(),
                nn.Linear(64, 2),
            )

        def forward(self, x):
            return self.classifier(self.features(x))

    # 教学数据：比较左上角和右下角区域的平均值来生成标签。
    torch.manual_seed(42)
    x = torch.randn(280, 3, 32, 32)
    left_top = x[:, :, :16, :16].mean(dim=(1, 2, 3))
    right_bottom = x[:, :, 16:, 16:].mean(dim=(1, 2, 3))
    y = (left_top > right_bottom).long()

    model = TeachingAlexNet()
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
    print("=== AlexNet Scratch ===")
    print(f"训练准确率={acc:.4f}")


if __name__ == "__main__":
    main()
