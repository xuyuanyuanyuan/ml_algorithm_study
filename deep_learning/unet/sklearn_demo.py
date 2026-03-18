"""U-Net（PyTorch 极简版）: 小型分割网络结构演示。"""


def main():
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
    except ImportError:
        print("缺少依赖，请安装: pip install torch")
        return

    class TinyUNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc1 = nn.Sequential(nn.Conv2d(1, 8, 3, padding=1), nn.ReLU())
            self.pool = nn.MaxPool2d(2)
            self.enc2 = nn.Sequential(nn.Conv2d(8, 16, 3, padding=1), nn.ReLU())
            self.up = nn.Upsample(scale_factor=2, mode="nearest")
            self.dec = nn.Sequential(nn.Conv2d(24, 8, 3, padding=1), nn.ReLU(), nn.Conv2d(8, 1, 1))

        def forward(self, x):
            e1 = self.enc1(x)
            e2 = self.enc2(self.pool(e1))
            u = self.up(e2)
            h = torch.cat([u, e1], dim=1)
            return self.dec(h)

    torch.manual_seed(42)
    x = torch.randn(32, 1, 32, 32)
    y = (x > 0).float()
    model = TinyUNet()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for _ in range(80):
        logits = model(x)
        loss = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("=== U-Net Demo ===")
    print(f"训练损失={loss.item():.6f}")


if __name__ == "__main__":
    main()
