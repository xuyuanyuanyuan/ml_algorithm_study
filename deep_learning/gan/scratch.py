"""GAN（手写教学版）: 生成器与判别器交替训练。"""


def main():
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
    except ImportError:
        print("缺少依赖，请安装: pip install torch")
        return

    class Generator(nn.Module):
        def __init__(self, z_dim=4):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(z_dim, 16), nn.ReLU(), nn.Linear(16, 1))

        def forward(self, z):
            return self.net(z)

    class Discriminator(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 1), nn.Sigmoid())

        def forward(self, x):
            return self.net(x)

    torch.manual_seed(42)
    g, d = Generator(), Discriminator()
    g_opt = optim.Adam(g.parameters(), lr=1e-3)
    d_opt = optim.Adam(d.parameters(), lr=1e-3)
    bce = nn.BCELoss()

    for _ in range(250):
        real = torch.randn(64, 1) * 0.7 + 1.5
        z = torch.randn(64, 4)
        fake = g(z).detach()
        d_loss = bce(d(real), torch.ones(64, 1)) + bce(d(fake), torch.zeros(64, 1))
        d_opt.zero_grad(); d_loss.backward(); d_opt.step()

        z = torch.randn(64, 4)
        fake = g(z)
        g_loss = bce(d(fake), torch.ones(64, 1))
        g_opt.zero_grad(); g_loss.backward(); g_opt.step()

    mean_fake = g(torch.randn(1000, 4)).mean().item()
    print("=== GAN Scratch ===")
    print(f"生成样本均值={mean_fake:.3f}")


if __name__ == "__main__":
    main()
