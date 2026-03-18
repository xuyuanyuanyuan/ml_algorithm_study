"""AutoEncoder（手写教学极简版）: 两层编码器 + 两层解码器。"""


def main():
    # 关键参数: latent_dim 表示压缩后的维度。
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
    except ImportError:
        print("缺少依赖，请安装: pip install torch")
        return

    class TinyAutoEncoder(nn.Module):
        def __init__(self, input_dim=16, latent_dim=2):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 8), nn.ReLU(), nn.Linear(8, latent_dim)
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, 8), nn.ReLU(), nn.Linear(8, input_dim)
            )

        def forward(self, x):
            z = self.encoder(x)
            return self.decoder(z), z

    torch.manual_seed(42)
    x = torch.randn(300, 16)
    model = TinyAutoEncoder(input_dim=16, latent_dim=2)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    for epoch in range(1, 151):
        recon, _ = model(x)
        loss = criterion(recon, x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 50 == 0:
            print(f"第 {epoch} 轮, loss={loss.item():.6f}")

    _, z = model(x)
    print("=== AutoEncoder Scratch ===")
    print(f"最终损失: {loss.item():.6f}, 压缩后形状: {tuple(z.shape)}")


if __name__ == "__main__":
    main()
