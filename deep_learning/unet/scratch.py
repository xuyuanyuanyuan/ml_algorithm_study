"""U-Net（手写教学版）: 编码器-解码器 + 跳跃连接。"""


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
            self.down = nn.MaxPool2d(2)
            self.enc2 = nn.Sequential(nn.Conv2d(8, 16, 3, padding=1), nn.ReLU())
            self.up = nn.Upsample(scale_factor=2, mode="nearest")
            self.dec = nn.Sequential(nn.Conv2d(24, 8, 3, padding=1), nn.ReLU(), nn.Conv2d(8, 1, 1))

        def forward(self, x):
            e1 = self.enc1(x)
            e2 = self.enc2(self.down(e1))
            u = self.up(e2)
            h = torch.cat([u, e1], dim=1)
            return self.dec(h)

    # 构造简易分割数据: 输入噪声图，目标是“正值区域”掩码。
    torch.manual_seed(42)
    x = torch.randn(40, 1, 32, 32)
    y = (x > 0).float()

    model = TinyUNet()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, 101):
        logits = model(x)
        loss = criterion(logits, y)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        if epoch % 25 == 0:
            print(f"第 {epoch} 轮, loss={loss.item():.6f}")

    print("=== U-Net Scratch ===")
    print(f"最终损失={loss.item():.6f}")


if __name__ == "__main__":
    main()
