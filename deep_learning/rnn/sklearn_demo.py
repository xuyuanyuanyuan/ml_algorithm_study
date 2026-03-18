"""RNN（PyTorch 简单版）: 序列二分类演示。"""


def main():
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
    except ImportError:
        print("缺少依赖，请安装: pip install torch")
        return

    torch.manual_seed(42)
    x = torch.randn(300, 8, 4)  # (batch, seq_len, input_dim)
    y = (x.sum(dim=(1, 2)) > 0).long()

    rnn = nn.RNN(input_size=4, hidden_size=12, batch_first=True)
    fc = nn.Linear(12, 2)
    optimizer = optim.Adam(list(rnn.parameters()) + list(fc.parameters()), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for _ in range(120):
        out, _ = rnn(x)
        logits = fc(out[:, -1, :])
        loss = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    acc = (logits.argmax(dim=1) == y).float().mean().item()
    print("=== RNN Demo ===")
    print(f"训练准确率={acc:.4f}")


if __name__ == "__main__":
    main()
