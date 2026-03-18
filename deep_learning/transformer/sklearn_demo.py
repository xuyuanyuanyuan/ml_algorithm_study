"""Transformer（PyTorch 简单版）: 自注意力序列分类示例。"""


def main():
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
    except ImportError:
        print("缺少依赖，请安装: pip install torch")
        return

    torch.manual_seed(42)
    x = torch.randn(280, 12, 8)
    y = (x[:, :, 0].mean(dim=1) > 0).long()

    encoder_layer = nn.TransformerEncoderLayer(
        d_model=8, nhead=2, dim_feedforward=32, batch_first=True
    )
    encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
    head = nn.Linear(8, 2)
    optimizer = optim.Adam(list(encoder.parameters()) + list(head.parameters()), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for _ in range(120):
        h = encoder(x)
        logits = head(h[:, -1, :])
        loss = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    acc = (logits.argmax(dim=1) == y).float().mean().item()
    print("=== Transformer Demo ===")
    print(f"训练准确率={acc:.4f}")


if __name__ == "__main__":
    main()
