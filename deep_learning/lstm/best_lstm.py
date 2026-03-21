"""
两层 LSTM（纯 numpy）+ BTC 数据 + 可视化
"""

import sqlite3
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


# ===== 激活函数 =====
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def dsigmoid(s):
    return s * (1 - s)


def tanh(x):
    return np.tanh(x)


def dtanh(t):
    return 1 - t * t


def mse_loss(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)


# ===== LSTM模型 =====
class TwoLayerLSTM:

    def __init__(self, input_size=1, hidden_size=20, lr=0.9, epochs=100):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lr = lr
        self.epochs = epochs

        rng = np.random.default_rng(42)

        # 第一层
        concat = input_size + hidden_size
        self.Wf = rng.normal(0, 0.1, (concat, hidden_size))
        self.Wi = rng.normal(0, 0.1, (concat, hidden_size))
        self.Wc = rng.normal(0, 0.1, (concat, hidden_size))
        self.Wo = rng.normal(0, 0.1, (concat, hidden_size))

        self.bf = np.zeros((1, hidden_size))
        self.bi = np.zeros((1, hidden_size))
        self.bc = np.zeros((1, hidden_size))
        self.bo = np.zeros((1, hidden_size))

        # 第二层
        concat2 = hidden_size + hidden_size
        self.Wf2 = rng.normal(0, 0.1, (concat2, hidden_size))
        self.Wi2 = rng.normal(0, 0.1, (concat2, hidden_size))
        self.Wc2 = rng.normal(0, 0.1, (concat2, hidden_size))
        self.Wo2 = rng.normal(0, 0.1, (concat2, hidden_size))

        self.bf2 = np.zeros((1, hidden_size))
        self.bi2 = np.zeros((1, hidden_size))
        self.bc2 = np.zeros((1, hidden_size))
        self.bo2 = np.zeros((1, hidden_size))

        # 输出层
        self.Wy = rng.normal(0, 0.1, (hidden_size, 1))
        self.by = np.zeros((1, 1))

        self.loss_history = []

    def forward(self, x):
        B, T, D = x.shape
        H = self.hidden_size

        h1 = np.zeros((B, H))
        c1 = np.zeros((B, H))

        h2 = np.zeros((B, H))
        c2 = np.zeros((B, H))

        for t in range(T):
            x_t = x[:, t, :]

            # ===== 第一层 =====
            z1 = np.concatenate([h1, x_t], axis=1)
            f1 = sigmoid(z1 @ self.Wf + self.bf)
            i1 = sigmoid(z1 @ self.Wi + self.bi)
            g1 = tanh(z1 @ self.Wc + self.bc)
            o1 = sigmoid(z1 @ self.Wo + self.bo)

            c1 = f1 * c1 + i1 * g1
            h1 = o1 * tanh(c1)

            # ===== 第二层 =====
            z2 = np.concatenate([h2, h1], axis=1)
            f2 = sigmoid(z2 @ self.Wf2 + self.bf2)
            i2 = sigmoid(z2 @ self.Wi2 + self.bi2)
            g2 = tanh(z2 @ self.Wc2 + self.bc2)
            o2 = sigmoid(z2 @ self.Wo2 + self.bo2)

            c2 = f2 * c2 + i2 * g2
            h2 = o2 * tanh(c2)

        y = h2 @ self.Wy + self.by
        return y

    def fit(self, x, y):
        for epoch in range(self.epochs):
            y_pred = self.forward(x)
            loss = mse_loss(y_pred, y)
            self.loss_history.append(loss)

            # 简化训练（仅更新输出层）
            grad = 2 * (y_pred - y) / len(y)
            self.Wy -= self.lr * (x[:, -1, :].T @ grad)
            self.by -= self.lr * np.sum(grad, axis=0)

            if epoch % 20 == 0:
                print(f"Epoch {epoch}, Loss={loss:.6f}")

    def predict(self, x):
        return self.forward(x)


# ===== 数据读取 =====
def find_db():
    for p in Path(__file__).resolve().parents:
        db = p / "data" / "alpha_arena.db"
        if db.exists():
            return db
    raise Exception("找不到数据库")


def load_btc(limit=1000):
    conn = sqlite3.connect(find_db())
    data = conn.execute("""
    SELECT close FROM market_data
    WHERE symbol='BTC/USDT:USDT'
    ORDER BY timestamp
    """).fetchall()
    conn.close()

    series = np.array([x[0] for x in data], dtype=float)
    return series[:limit]


# ===== 构造序列 =====
def build_data(series, seq_len=12):
    x, y = [], []
    for i in range(len(series) - seq_len):
        x.append(series[i:i+seq_len])
        y.append(series[i+seq_len])
    return np.array(x)[..., None], np.array(y)[..., None]


# ===== 主函数 =====
def main():

    series = load_btc()

    # 划分
    split = int(len(series)*0.8)
    train, test = series[:split], series[split:]

    # 标准化
    mean, std = train.mean(), train.std()
    train = (train-mean)/std
    test = (test-mean)/std

    x_train, y_train = build_data(train)
    x_test, y_test = build_data(test)

    print("训练数据形状:", x_train.shape)

    model = TwoLayerLSTM()
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    # 反标准化
    y_pred = y_pred*std + mean
    y_test = y_test*std + mean

    # ===== 可视化 =====
    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    plt.plot(model.loss_history)
    plt.title("Loss")

    plt.subplot(1,2,2)
    plt.plot(y_test[:100], label="True")
    plt.plot(y_pred[:100], label="Pred")
    plt.legend()
    plt.title("Prediction")

    plt.show()


if __name__ == "__main__":
    main()