"""LSTM（手写教学版）: 理解门控循环结构的最小示例。"""


def main():
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
    except ImportError:
        print("缺少依赖，请安装: pip install torch")
        return

    class SimpleLSTMClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(input_size=4, hidden_size=12, batch_first=True)
            self.fc = nn.Linear(12, 2)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :])

    x = torch.randn(260, 9, 4)
    y = (x[:, :, 2].mean(dim=1) > 0).long()
    model = SimpleLSTMClassifier()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for _ in range(140):
        logits = model(x)
        loss = criterion(logits, y)
        optimizer.zero_grad(); loss.backward(); optimizer.step()

    acc = (model(x).argmax(dim=1) == y).float().mean().item()
    print("=== LSTM Scratch ===")
    print(f"训练准确率={acc:.4f}")


if __name__ == "__main__":
    main()
