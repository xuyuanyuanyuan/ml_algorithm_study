"""VGG（手写教学简化版）: 强调“重复卷积块”的搭建思路。"""


def main():
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
    except ImportError:
        print("缺少依赖，请安装: pip install torch")
        return

    def vgg_block(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

    class MiniVGGScratch(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                vgg_block(3, 16),
                vgg_block(16, 32),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(32 * 8 * 8, 64),
                nn.ReLU(),
                nn.Linear(64, 2),
            )

        def forward(self, x):
            return self.classifier(self.features(x))

    # 教学数据：根据两条对角区域均值差构造标签。
    torch.manual_seed(42)
    x = torch.randn(300, 3, 32, 32)
    diag_a = x[:, :, :16, :16].mean(dim=(1, 2, 3))
    diag_b = x[:, :, :16, 16:].mean(dim=(1, 2, 3))
    y = (diag_a > diag_b).long()

    model = MiniVGGScratch()
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
    print("=== VGG Scratch ===")
    print(f"训练准确率={acc:.4f}")


if __name__ == "__main__":
    main()
