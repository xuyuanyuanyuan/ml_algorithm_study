"""LSTM（纯手写 numpy 时间序列预测教学版）
任务：
- 输入过去 seq_len 个时间点
- 预测下一个时间点的值
- 纯 numpy 实现单层 LSTM
"""

import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def dsigmoid_from_output(s):
    return s * (1.0 - s)


def tanh(x):
    return np.tanh(x)


def dtanh_from_output(t):
    return 1.0 - t * t


def mse_loss(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)


class LSTMScratchRegressor:
    """
    单层 LSTM 时间序列回归器（教学版）
    输入:  x shape = (B, T, D)
    输出:  y_pred shape = (B, 1)
    """

    def __init__(self, input_size=1, hidden_size=8, learning_rate=0.01, epochs=200, seed=42):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.epochs = epochs

        rng = np.random.default_rng(seed)
        concat_size = input_size + hidden_size

        # 四个门的参数
        self.Wf = rng.normal(0, 0.1, size=(concat_size, hidden_size))
        self.bf = np.zeros((1, hidden_size))

        self.Wi = rng.normal(0, 0.1, size=(concat_size, hidden_size))
        self.bi = np.zeros((1, hidden_size))

        self.Wc = rng.normal(0, 0.1, size=(concat_size, hidden_size))
        self.bc = np.zeros((1, hidden_size))

        self.Wo = rng.normal(0, 0.1, size=(concat_size, hidden_size))
        self.bo = np.zeros((1, hidden_size))

        # 输出层：最后隐藏状态 -> 一个实数
        self.Wy = rng.normal(0, 0.1, size=(hidden_size, 1))
        self.by = np.zeros((1, 1))

        self.loss_history = []

    def forward(self, x):
        """
        前向传播
        x: (B, T, D)
        """
        B, T, D = x.shape
        H = self.hidden_size

        h_prev = np.zeros((B, H))
        c_prev = np.zeros((B, H))

        seq_cache = []

        for t in range(T):
            x_t = x[:, t, :]  # (B, D)

            # 拼接 [h_{t-1}, x_t]
            z_t = np.concatenate([h_prev, x_t], axis=1)  # (B, H+D)

            f_t = sigmoid(z_t @ self.Wf + self.bf)   # (B, H)
            i_t = sigmoid(z_t @ self.Wi + self.bi)   # (B, H)
            g_t = tanh(z_t @ self.Wc + self.bc)      # (B, H)
            o_t = sigmoid(z_t @ self.Wo + self.bo)   # (B, H)

            c_t = f_t * c_prev + i_t * g_t           # (B, H)
            tanh_c_t = tanh(c_t)
            h_t = o_t * tanh_c_t                     # (B, H)

            seq_cache.append({
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

        h_last = h_prev  # (B, H)
        y_pred = h_last @ self.Wy + self.by  # (B, 1)

        cache = {
            "seq_cache": seq_cache,
            "h_last": h_last,
            "x": x,
            "y_pred": y_pred,
        }
        return y_pred, cache

    def backward(self, cache, y_true):
        """
        反向传播（BPTT，教学简化版）
        y_true: (B, 1)
        """
        seq_cache = cache["seq_cache"]
        h_last = cache["h_last"]
        y_pred = cache["y_pred"]

        B = y_true.shape[0]
        H = self.hidden_size
        T = len(seq_cache)

        # y_pred shape: (B,1)
        # MSE = mean((y_pred - y_true)^2)
        dy = 2.0 * (y_pred - y_true) / B  # (B,1)

        # 输出层梯度
        dWy = h_last.T @ dy                # (H,B)@(B,1)->(H,1)
        dby = np.sum(dy, axis=0, keepdims=True)  # (1,1)

        dh_next = dy @ self.Wy.T           # (B,1)@(1,H)->(B,H)
        dc_next = np.zeros((B, H))

        # 初始化 LSTM 参数梯度
        dWf = np.zeros_like(self.Wf)
        dbf = np.zeros_like(self.bf)

        dWi = np.zeros_like(self.Wi)
        dbi = np.zeros_like(self.bi)

        dWc = np.zeros_like(self.Wc)
        dbc = np.zeros_like(self.bc)

        dWo = np.zeros_like(self.Wo)
        dbo = np.zeros_like(self.bo)

        for t in reversed(range(T)):
            step = seq_cache[t]

            z_t = step["z_t"]
            h_prev = step["h_prev"]
            c_prev = step["c_prev"]
            f_t = step["f_t"]
            i_t = step["i_t"]
            g_t = step["g_t"]
            o_t = step["o_t"]
            c_t = step["c_t"]
            tanh_c_t = step["tanh_c_t"]

            dh = dh_next

            # h_t = o_t * tanh(c_t)
            do = dh * tanh_c_t
            do_raw = do * dsigmoid_from_output(o_t)

            dc = dh * o_t * dtanh_from_output(tanh_c_t) + dc_next

            # c_t = f_t * c_prev + i_t * g_t
            df = dc * c_prev
            di = dc * g_t
            dg = dc * i_t
            dc_prev = dc * f_t

            df_raw = df * dsigmoid_from_output(f_t)
            di_raw = di * dsigmoid_from_output(i_t)
            dg_raw = dg * dtanh_from_output(g_t)

            dWf += z_t.T @ df_raw
            dbf += np.sum(df_raw, axis=0, keepdims=True)

            dWi += z_t.T @ di_raw
            dbi += np.sum(di_raw, axis=0, keepdims=True)

            dWc += z_t.T @ dg_raw
            dbc += np.sum(dg_raw, axis=0, keepdims=True)

            dWo += z_t.T @ do_raw
            dbo += np.sum(do_raw, axis=0, keepdims=True)

            dz = (
                df_raw @ self.Wf.T
                + di_raw @ self.Wi.T
                + dg_raw @ self.Wc.T
                + do_raw @ self.Wo.T
            )  # (B, H+D)

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

    def fit(self, x, y, print_every=20):
        """
        x: (B, T, D)
        y: (B, 1)
        """
        for epoch in range(1, self.epochs + 1):
            y_pred, cache = self.forward(x)
            loss = mse_loss(y_pred, y)
            grads = self.backward(cache, y)
            self.update_parameters(grads)

            self.loss_history.append(loss)

            if epoch % print_every == 0:
                print(f"第 {epoch:4d} 轮: loss={loss:.6f}")

    def predict(self, x):
        y_pred, _ = self.forward(x)
        return y_pred


def build_sine_data(seq_len=12, total_points=320):
    """
    用正弦波构造单变量时间序列数据。
    任务：
    用前 seq_len 个点预测下一个点
    """
    t = np.linspace(0, 20 * np.pi, total_points)
    series = np.sin(t)

    x_list = []
    y_list = []

    for i in range(len(series) - seq_len):
        x_window = series[i:i + seq_len]          # 长度 seq_len
        y_target = series[i + seq_len]            # 下一个点

        x_list.append(x_window.reshape(seq_len, 1))  # (T,1)
        y_list.append([y_target])                    # (1,)

    x = np.array(x_list, dtype=float)  # (N, T, 1)
    y = np.array(y_list, dtype=float)  # (N, 1)
    return x, y


def train_test_split_numpy(x, y, test_ratio=0.2):
    split = int(len(x) * (1 - test_ratio))
    return x[:split], x[split:], y[:split], y[split:]


def main():
    # 1. 构造时间序列数据
    x, y = build_sine_data(seq_len=12, total_points=320)
    x_train, x_test, y_train, y_test = train_test_split_numpy(x, y, test_ratio=0.2)

    # 2. 创建模型
    model = LSTMScratchRegressor(
        input_size=1,
        hidden_size=8,
        learning_rate=0.01,
        epochs=200,
        seed=42,
    )

    # 3. 训练
    model.fit(x_train, y_train, print_every=20)

    # 4. 测试
    y_pred = model.predict(x_test)
    test_loss = mse_loss(y_pred, y_test)

    print("=== LSTM（纯手写 numpy 时间序列预测版）===")
    print(f"测试集 MSE = {test_loss:.6f}")

    # 打印前5个预测值看看
    print("前5个真实值 vs 预测值：")
    for i in range(5):
        print(f"true={y_test[i,0]:.4f}, pred={y_pred[i,0]:.4f}")


if __name__ == "__main__":
    main()