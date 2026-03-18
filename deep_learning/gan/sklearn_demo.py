"""GAN（PyTorch 简单版）: 1D 高斯分布生成演示。"""


def main():
    # 关键参数: z_dim 噪声维度。
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
    except ImportError:
        print("缺少依赖，请安装: pip install torch")
        return

    torch.manual_seed(42)
    z_dim = 5
    G = nn.Sequential(nn.Linear(z_dim, 16), nn.ReLU(), nn.Linear(16, 1))
    D = nn.Sequential(nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 1), nn.Sigmoid())
    g_opt = optim.Adam(G.parameters(), lr=1e-3)
    d_opt = optim.Adam(D.parameters(), lr=1e-3)
    bce = nn.BCELoss()

    for _ in range(200):
        real = torch.randn(64, 1) * 0.8 + 2.0
        z = torch.randn(64, z_dim)
        fake = G(z).detach()
        d_loss = bce(D(real), torch.ones(64, 1)) + bce(D(fake), torch.zeros(64, 1))
        d_opt.zero_grad(); d_loss.backward(); d_opt.step()

        z = torch.randn(64, z_dim)
        fake = G(z)
        g_loss = bce(D(fake), torch.ones(64, 1))
        g_opt.zero_grad(); g_loss.backward(); g_opt.step()

    sample_mean = G(torch.randn(1000, z_dim)).mean().item()
    print("=== GAN Demo ===")
    print(f"生成样本均值(目标约2.0): {sample_mean:.3f}")


if __name__ == "__main__":
    main()
