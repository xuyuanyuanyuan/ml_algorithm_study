"""CNN（PyTorch 简单版）: 在 digits 小数据上做分类。"""


def main():
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from sklearn.datasets import load_digits
    except ImportError:
        print("缺少依赖，请安装: pip install torch scikit-learn")
        return

    data = load_digits()
    x = torch.tensor(data.images, dtype=torch.float32).unsqueeze(1) / 16.0
    y = torch.tensor(data.target, dtype=torch.long)

    model = nn.Sequential(
        nn.Conv2d(1, 8, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(8 * 8 * 8, 10),
    )
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for _ in range(50):
        logits = model(x)
        loss = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    acc = (model(x).argmax(dim=1) == y).float().mean().item()
    print("=== CNN Demo ===")
    print(f"训练准确率={acc:.4f}")


if __name__ == "__main__":
    main()
