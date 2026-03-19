"""LSTM（ numpy 教学版）："""

import numpy as np


def sigmoid(x):
    """sigmoid 激活函数。"""
    return 1.0 / (1.0 + np.exp(-x))


def dsigmoid_from_output(s):
    """已知 sigmoid 输出 s，求导数。"""
    return s * (1.0 - s)


def tanh(x):
    """tanh 激活函数。"""
    return np.tanh(x)


def dtanh_from_output(t):
    """已知 tanh 输出 t，求导数。"""
    return 1.0 - t * t


def softmax(x):
    """对二维数组做 softmax，按行归一化。"""
    x_shift = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x_shift)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def cross_entropy_loss(probs, y_true):
    """交叉熵损失。
    probs: shape (batch_size, num_classes)
    y_true: shape (batch_size,)
    """
    batch_size = probs.shape[0]
    eps = 1e-12
    correct_probs = probs[np.arange(batch_size), y_true]
    return -np.mean(np.log(correct_probs + eps))


def one_hot(y, num_classes):
    """把标签转成 one-hot。"""
    out = np.zeros((len(y), num_classes))
    out[np.arange(len(y)), y] = 1.0
    return out


class LSTMScratchClassifier:
    """
    纯手写 LSTM 二分类器（教学版）。

    结构：
    输入序列 x -> LSTM -> 最后一个时间步的隐藏状态 h_T -> 线性层 -> softmax

    记号：
    - batch_size = B
    - seq_len = T
    - input_size = D
    - hidden_size = H
    - num_classes = C
    """

    def __init__(self, input_size=4, hidden_size=12, num_classes=2, learning_rate=0.01, epochs=200, seed=42):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.epochs = epochs

        rng = np.random.default_rng(seed)

        # LSTM 部分参数
        # 这里把 h_{t-1} 和 x_t 拼接，拼接后维度 = hidden_size + input_size
        concat_size = hidden_size + input_size

        # 四个门各自一套参数
        self.Wf = rng.normal(0, 0.1, size=(concat_size, hidden_size))
        self.bf = np.zeros((1, hidden_size))

        self.Wi = rng.normal(0, 0.1, size=(concat_size, hidden_size))
        self.bi = np.zeros((1, hidden_size))

        self.Wc = rng.normal(0, 0.1, size=(concat_size, hidden_size))
        self.bc = np.zeros((1, hidden_size))

        self.Wo = rng.normal(0, 0.1, size=(concat_size, hidden_size))
        self.bo = np.zeros((1, hidden_size))

        # 最后分类层参数：h_T -> logits
        self.Wy = rng.normal(0, 0.1, size=(hidden_size, num_classes))
        self.by = np.zeros((1, num_classes))

        self.loss_history = []

    def forward(self, x):
        """
        前向传播。

        x: shape (B, T, D)
        返回：
        - probs: shape (B, C)
        - cache: 保存中间结果，供反向传播使用
        """
        B, T, D = x.shape
        H = self.hidden_size

        # 初始隐藏状态 h0 和细胞状态 c0
        h_prev = np.zeros((B, H))
        c_prev = np.zeros((B, H))

        # 保存每个时间步中间变量，供 BPTT 使用
        caches = []

        for t in range(T):
            x_t = x[:, t, :]  # shape (B, D)

            # 拼接 [h_{t-1}, x_t]
            z_t = np.concatenate([h_prev, x_t], axis=1)  # shape (B, H + D)

            # 四个门
            f_t = sigmoid(z_t @ self.Wf + self.bf)   # 遗忘门
            i_t = sigmoid(z_t @ self.Wi + self.bi)   # 输入门
            g_t = tanh(z_t @ self.Wc + self.bc)      # 候选记忆
            o_t = sigmoid(z_t @ self.Wo + self.bo)   # 输出门

            # 更新细胞状态 c_t
            c_t = f_t * c_prev + i_t * g_t

            # 更新隐藏状态 h_t
            tanh_c_t = tanh(c_t)
            h_t = o_t * tanh_c_t

            caches.append({
                "x_t": x_t,
                "z_t": z_t,
                "h_prev": h_prev,
                "c_prev": c_prev,
                "f_t": f_t,
                "i_t": i_t,
                "g_t": g_t,
                "o_t": o_t,
                "c_t": c_t,
                "tanh_c_t": tanh_c_t,
                "h_t": h_t,
            })

            h_prev = h_t
            c_prev = c_t

        # 取最后一个时间步的隐藏状态做分类
        h_last = h_prev  # shape (B, H)

        logits = h_last @ self.Wy + self.by  # shape (B, C)
        probs = softmax(logits)

        cache = {
            "seq_caches": caches,
            "h_last": h_last,
            "logits": logits,
            "probs": probs,
            "x": x,
        }
        return probs, cache

    def backward(self, cache, y_true):
        """
        反向传播（BPTT，教学版）。

        y_true: shape (B,)
        返回所有参数梯度
        """
        seq_caches = cache["seq_caches"]
        probs = cache["probs"]
        h_last = cache["h_last"]

        B = probs.shape[0]
        T = len(seq_caches)
        H = self.hidden_size
        D = self.input_size

        # ===== 1. 分类层梯度 =====
        y_onehot = one_hot(y_true, self.num_classes)  # shape (B, C)

        # softmax + cross entropy 的经典梯度
        dlogits = (probs - y_onehot) / B  # shape (B, C)

        dWy = h_last.T @ dlogits                # (H, B) @ (B, C) -> (H, C)
        dby = np.sum(dlogits, axis=0, keepdims=True)  # (1, C)

        # 来自最后分类层对 h_last 的梯度
        dh_next = dlogits @ self.Wy.T  # shape (B, H)

        # 初始 dc_next = 0
        dc_next = np.zeros((B, H))

        # ===== 2. LSTM 参数梯度初始化 =====
        dWf = np.zeros_like(self.Wf)
        dbf = np.zeros_like(self.bf)

        dWi = np.zeros_like(self.Wi)
        dbi = np.zeros_like(self.bi)

        dWc = np.zeros_like(self.Wc)
        dbc = np.zeros_like(self.bc)

        dWo = np.zeros_like(self.Wo)
        dbo = np.zeros_like(self.bo)

        # ===== 3. 从最后一个时间步往前做 BPTT =====
        for t in reversed(range(T)):
            step = seq_caches[t]

            z_t = step["z_t"]               # (B, H + D)
            h_prev = step["h_prev"]         # (B, H)
            c_prev = step["c_prev"]         # (B, H)
            f_t = step["f_t"]               # (B, H)
            i_t = step["i_t"]               # (B, H)
            g_t = step["g_t"]               # (B, H)
            o_t = step["o_t"]               # (B, H)
            c_t = step["c_t"]               # (B, H)
            tanh_c_t = step["tanh_c_t"]     # (B, H)

            # 当前时刻 h_t = o_t * tanh(c_t)
            # 所以：
            # dh = 来自未来时间步 + 来自分类层（最后一步）
            dh = dh_next

            # 对输出门 o_t 的梯度
            do = dh * tanh_c_t
            do_raw = do * dsigmoid_from_output(o_t)

            # 对 c_t 的梯度
            dc = dh * o_t * dtanh_from_output(tanh_c_t) + dc_next

            # c_t = f_t * c_prev + i_t * g_t
            df = dc * c_prev
            di = dc * g_t
            dg = dc * i_t
            dc_prev = dc * f_t

            df_raw = df * dsigmoid_from_output(f_t)
            di_raw = di * dsigmoid_from_output(i_t)
            dg_raw = dg * dtanh_from_output(g_t)

            # 各门参数梯度
            dWf += z_t.T @ df_raw
            dbf += np.sum(df_raw, axis=0, keepdims=True)

            dWi += z_t.T @ di_raw
            dbi += np.sum(di_raw, axis=0, keepdims=True)

            dWc += z_t.T @ dg_raw
            dbc += np.sum(dg_raw, axis=0, keepdims=True)

            dWo += z_t.T @ do_raw
            dbo += np.sum(do_raw, axis=0, keepdims=True)

            # z_t = [h_prev, x_t]
            # 所以 dz_t 要从四个门反向加起来
            dz = (
                df_raw @ self.Wf.T
                + di_raw @ self.Wi.T
                + dg_raw @ self.Wc.T
                + do_raw @ self.Wo.T
            )  # shape (B, H + D)

            # dz 的前 H 维对应 h_prev，后 D 维对应 x_t
            dh_next = dz[:, :H]
            dc_next = dc_prev

        grads = {
            "Wf": dWf, "bf": dbf,
            "Wi": dWi, "bi": dbi,
            "Wc": dWc, "bc": dbc,
            "Wo": dWo, "bo": dbo,
            "Wy": dWy, "by": dby,
        }
        return grads

    def update_parameters(self, grads):
        """梯度下降更新参数。"""
        lr = self.learning_rate

        self.Wf -= lr * grads["Wf"]
        self.bf -= lr * grads["bf"]

        self.Wi -= lr * grads["Wi"]
        self.bi -= lr * grads["bi"]

        self.Wc -= lr * grads["Wc"]
        self.bc -= lr * grads["bc"]

        self.Wo -= lr * grads["Wo"]
        self.bo -= lr * grads["bo"]

        self.Wy -= lr * grads["Wy"]
        self.by -= lr * grads["by"]

    def predict(self, x):
        """预测类别。"""
        probs, _ = self.forward(x)
        return np.argmax(probs, axis=1)

    def fit(self, x, y, print_every=20):
        """训练。"""
        for epoch in range(1, self.epochs + 1):
            probs, cache = self.forward(x)
            loss = cross_entropy_loss(probs, y)
            grads = self.backward(cache, y)
            self.update_parameters(grads)

            self.loss_history.append(loss)

            if epoch % print_every == 0:
                pred = np.argmax(probs, axis=1)
                acc = np.mean(pred == y)
                print(f"第 {epoch:4d} 轮: loss={loss:.6f}, acc={acc:.4f}")


def build_demo_sequence_data(n_samples=240, seq_len=9, input_size=4, seed=42):
    """
    构造一个简单序列分类数据集。
    规则：
    - 如果一个序列中第3维特征（索引2）的平均值 > 0，则标签为1
    - 否则标签为0

    返回：
    x: shape (N, T, D)
    y: shape (N,)
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, size=(n_samples, seq_len, input_size))
    y = (x[:, :, 2].mean(axis=1) > 0).astype(int)
    return x, y


def train_test_split_numpy(x, y, test_ratio=0.2, seed=42):
    """简单划分训练集和测试集。"""
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(x))
    x = x[indices]
    y = y[indices]
    split = int(len(x) * (1 - test_ratio))
    return x[:split], x[split:], y[:split], y[split:]


def main():
    # 1. 构造一个最小序列分类数据集
    x, y = build_demo_sequence_data(n_samples=260, seq_len=9, input_size=4, seed=42)
    x_train, x_test, y_train, y_test = train_test_split_numpy(x, y, test_ratio=0.2, seed=42)

    # 2. 创建模型
    model = LSTMScratchClassifier(
        input_size=4,
        hidden_size=12,
        num_classes=2,
        learning_rate=0.01,
        epochs=200,
        seed=42,
    )

    # 3. 训练
    model.fit(x_train, y_train, print_every=20)

    # 4. 测试
    y_pred = model.predict(x_test)
    test_acc = np.mean(y_pred == y_test)

    print("=== LSTM（纯手写 numpy 教学版）===")
    print(f"测试集准确率 = {test_acc:.4f}")


if __name__ == "__main__":
    main()