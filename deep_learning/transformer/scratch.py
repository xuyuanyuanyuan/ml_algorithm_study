"""Transformer（手写教学版）: 自注意力 + 编码器结构模板。"""


def main():
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
    except ImportError:
        print("缺少依赖，请安装: pip install torch")
        return

    class TinyTransformerClassifier(nn.Module):
        def __init__(self, d_model=8, nhead=2):
            super().__init__()
            layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=32, batch_first=True
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=2)
            self.fc = nn.Linear(d_model, 2)

        def forward(self, x):
            h = self.encoder(x)
            return self.fc(h[:, -1, :])

    x = torch.randn(240, 10, 8)
    y = (x[:, :, 0].sum(dim=1) > 0).long()
    model = TinyTransformerClassifier(d_model=8, nhead=2)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for _ in range(120):
        logits = model(x)
        loss = criterion(logits, y)
        optimizer.zero_grad(); loss.backward(); optimizer.step()

    acc = (model(x).argmax(dim=1) == y).float().mean().item()
    print("=== Transformer Scratch ===")
    print(f"训练准确率={acc:.4f}")


if __name__ == "__main__":
    main()
