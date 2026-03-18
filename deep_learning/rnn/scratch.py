"""RNN（手写教学版）: 用循环单元处理时间序列。"""


def main():
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
    except ImportError:
        print("缺少依赖，请安装: pip install torch")
        return

    class SimpleRNNClassifier(nn.Module):
        def __init__(self, input_dim=3, hidden_dim=10, num_classes=2):
            super().__init__()
            self.rnn = nn.RNN(input_dim, hidden_dim, batch_first=True)
            self.fc = nn.Linear(hidden_dim, num_classes)

        def forward(self, x):
            out, _ = self.rnn(x)
            return self.fc(out[:, -1, :])

    x = torch.randn(240, 7, 3)
    y = (x[:, :, 0].sum(dim=1) > 0).long()
    model = SimpleRNNClassifier()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for _ in range(120):
        logits = model(x)
        loss = criterion(logits, y)
        optimizer.zero_grad(); loss.backward(); optimizer.step()

    acc = (model(x).argmax(dim=1) == y).float().mean().item()
    print("=== RNN Scratch ===")
    print(f"训练准确率={acc:.4f}")


if __name__ == "__main__":
    main()
