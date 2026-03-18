"""LSTM（PyTorch 简单版）: 处理序列依赖关系。"""


def main():
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
    except ImportError:
        print("缺少依赖，请安装: pip install torch")
        return

    torch.manual_seed(42)
    x = torch.randn(320, 10, 5)
    y = (x[:, :, 0].sum(dim=1) > 0).long()

    lstm = nn.LSTM(input_size=5, hidden_size=16, batch_first=True)
    fc = nn.Linear(16, 2)
    optimizer = optim.Adam(list(lstm.parameters()) + list(fc.parameters()), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for _ in range(140):
        out, _ = lstm(x)
        logits = fc(out[:, -1, :])
        loss = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    acc = (logits.argmax(dim=1) == y).float().mean().item()
    print("=== LSTM Demo ===")
    print(f"训练准确率={acc:.4f}")


if __name__ == "__main__":
    main()
