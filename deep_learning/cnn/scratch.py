"""CNN（手写教学版）: 最小卷积网络训练流程模板。"""


def main():
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
    except ImportError:
        print("缺少依赖，请安装: pip install torch")
        return

    # 关键参数: 卷积核大小 kernel_size, 通道数 out_channels。
    class SimpleCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(1, 6, kernel_size=3, padding=1)
            self.act = nn.ReLU()
            self.fc = nn.Linear(6 * 8 * 8, 2)

        def forward(self, x):
            h = self.act(self.conv(x))
            h = h.view(x.size(0), -1)
            return self.fc(h)

    x = torch.randn(200, 1, 8, 8)
    y = (x.mean(dim=(1, 2, 3)) > 0).long()
    model = SimpleCNN()
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
    print("=== CNN Scratch ===")
    print(f"训练准确率={acc:.4f}")


if __name__ == "__main__":
    main()
