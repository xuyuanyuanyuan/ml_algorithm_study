"""Bidirectional RNN（PyTorch 简单版）: 双向序列建模二分类演示。"""


def main():
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from sklearn.model_selection import train_test_split
    except ImportError:
        print("缺少依赖，请安装: pip install torch scikit-learn")
        return

    class BiRNNClassifier(nn.Module):
        """双向 RNN：同时读取过去和未来上下文。"""

        def __init__(self, input_dim=6, hidden_dim=16, num_classes=2):
            super().__init__()
            self.rnn = nn.RNN(
                input_size=input_dim,
                hidden_size=hidden_dim,
                batch_first=True,
                bidirectional=True,
            )
            self.fc = nn.Linear(hidden_dim * 2, num_classes)

        def forward(self, x):
            out, _ = self.rnn(x)
            return self.fc(out[:, -1, :])

    torch.manual_seed(42)
    x = torch.randn(600, 12, 6)
    # 同时依赖前半段与后半段信息，体现双向建模场景。
    score = x[:, :6, 0].sum(dim=1) + x[:, 6:, 1].sum(dim=1)
    y = (score > 0).long()

    x_train_np, x_test_np, y_train_np, y_test_np = train_test_split(
        x.numpy(), y.numpy(), test_size=0.2, random_state=42, stratify=y.numpy()
    )
    x_train = torch.tensor(x_train_np, dtype=torch.float32)
    x_test = torch.tensor(x_test_np, dtype=torch.float32)
    y_train = torch.tensor(y_train_np, dtype=torch.long)
    y_test = torch.tensor(y_test_np, dtype=torch.long)

    model = BiRNNClassifier(input_dim=6, hidden_dim=18, num_classes=2)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for _ in range(120):
        logits = model(x_train)
        loss = criterion(logits, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        pred = model(x_test).argmax(dim=1)
        acc = (pred == y_test).float().mean().item()

    print("=== Bidirectional RNN Demo ===")
    print(f"测试准确率={acc:.4f}")


if __name__ == "__main__":
    main()
