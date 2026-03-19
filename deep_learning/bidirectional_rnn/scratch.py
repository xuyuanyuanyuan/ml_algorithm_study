"""Bidirectional RNN（手写教学简化版）: 最小序列输入示例。"""


def main():
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
    except ImportError:
        print("缺少依赖，请安装: pip install torch")
        return

    class SimpleBiRNN(nn.Module):
        """教学双向 RNN 分类器。"""

        def __init__(self, input_dim=4, hidden_dim=10, num_classes=2):
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
    x = torch.randn(320, 10, 4)
    y = ((x[:, :, 0].mean(dim=1) - x[:, :, 1].mean(dim=1)) > 0).long()

    model = SimpleBiRNN(input_dim=4, hidden_dim=12, num_classes=2)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, 121):
        logits = model(x)
        loss = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 40 == 0:
            print(f"第 {epoch} 轮, loss={loss.item():.6f}")

    # 最小序列输入示例：1 个样本、10 个时间步、每步 4 个特征。
    sample = torch.zeros(1, 10, 4)
    sample[:, :5, 0] = 1.0
    sample[:, 5:, 1] = -0.5

    with torch.no_grad():
        train_acc = (model(x).argmax(dim=1) == y).float().mean().item()
        sample_pred = int(model(sample).argmax(dim=1).item())

    print("=== Bidirectional RNN Scratch ===")
    print(f"训练准确率={train_acc:.4f}")
    print(f"示例序列预测类别={sample_pred}")


if __name__ == "__main__":
    main()
