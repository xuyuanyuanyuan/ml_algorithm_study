"""AutoEncoder（PyTorch 简单版）: 学习压缩表示。"""


def main():
    # 关键参数: latent_dim 是隐空间维度，越小压缩越强。
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
    except ImportError:
        print("缺少依赖，请安装: pip install torch")
        return

    torch.manual_seed(42)
    x = torch.randn(256, 20)

    model = nn.Sequential(
        nn.Linear(20, 10),
        nn.ReLU(),
        nn.Linear(10, 3),
        nn.ReLU(),
        nn.Linear(3, 10),
        nn.ReLU(),
        nn.Linear(10, 20),
    )
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for _ in range(120):
        recon = model(x)
        loss = criterion(recon, x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("=== AutoEncoder Demo (torch) ===")
    print(f"重构损失: {loss.item():.6f}")


if __name__ == "__main__":
    main()
