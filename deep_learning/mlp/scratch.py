"""MLP（手写教学版）：最简单两层神经网络（二分类）。"""


def _get_styled_pyplot():
    """获取统一风格的 pyplot。"""
    try:
        import sys
        from pathlib import Path

        project_root = None
        for parent in Path(__file__).resolve().parents:
            if (parent / "utils" / "plot_utils.py").exists():
                project_root = parent
                break
        if project_root is None:
            return None
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        from utils.plot_utils import get_styled_pyplot
    except Exception:
        return None
    return get_styled_pyplot()


class TwoLayerMLPScratch:
    """
    两层神经网络（输入层 -> 隐藏层 -> 输出层）。
    说明：
    - 隐藏层激活函数：Sigmoid
    - 输出层激活函数：Sigmoid
    - 损失函数：二元交叉熵
    """

    def __init__(self, input_size=2, hidden_size=8, learning_rate=0.1, epochs=4000, random_state=42):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.random_state = random_state

        self.w1 = None
        self.b1 = None
        self.w2 = None
        self.b2 = None
        self.loss_history = []

    def _init_parameters(self):
        """参数初始化。"""
        import numpy as np

        rng = np.random.default_rng(self.random_state)
        self.w1 = rng.normal(0, 0.6, size=(self.input_size, self.hidden_size))
        self.b1 = np.zeros((1, self.hidden_size))
        self.w2 = rng.normal(0, 0.6, size=(self.hidden_size, 1))
        self.b2 = np.zeros((1, 1))

    def _sigmoid(self, z):
        """Sigmoid 激活函数。"""
        import numpy as np

        return 1.0 / (1.0 + np.exp(-z))

    def _forward(self, x_data):
        """前向传播。"""
        z1 = x_data @ self.w1 + self.b1
        a1 = self._sigmoid(z1)
        z2 = a1 @ self.w2 + self.b2
        a2 = self._sigmoid(z2)
        return {"z1": z1, "a1": a1, "z2": z2, "a2": a2}

    def _binary_cross_entropy(self, y_true, y_prob):
        """二元交叉熵损失。"""
        import numpy as np

        eps = 1e-8
        y_true = y_true.reshape(-1, 1)
        loss = -np.mean(
            y_true * np.log(y_prob + eps)
            + (1 - y_true) * np.log(1 - y_prob + eps)
        )
        return float(loss)

    def _backward(self, x_data, y_true, cache):
        """反向传播并更新参数。"""
        import numpy as np

        m = x_data.shape[0]
        y_true = y_true.reshape(-1, 1)

        a1 = cache["a1"]
        a2 = cache["a2"]

        # 输出层梯度。
        dz2 = a2 - y_true
        dw2 = (a1.T @ dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m

        # 隐藏层梯度。
        dz1 = (dz2 @ self.w2.T) * a1 * (1 - a1)
        dw1 = (x_data.T @ dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m

        # 梯度下降更新参数。
        self.w2 -= self.learning_rate * dw2
        self.b2 -= self.learning_rate * db2
        self.w1 -= self.learning_rate * dw1
        self.b1 -= self.learning_rate * db1

    def fit(self, x_train, y_train, print_every=400):
        """训练模型。"""
        self._init_parameters()
        self.loss_history = []

        for epoch in range(1, self.epochs + 1):
            cache = self._forward(x_train)
            loss = self._binary_cross_entropy(y_train, cache["a2"])
            self._backward(x_train, y_train, cache)
            self.loss_history.append(loss)

            if epoch % print_every == 0:
                print(f"第 {epoch:4d} 轮，loss={loss:.6f}")

    def predict_proba(self, x_data):
        """输出类别 1 的概率。"""
        cache = self._forward(x_data)
        return cache["a2"].ravel()

    def predict(self, x_data, threshold=0.5):
        """按阈值输出类别标签。"""
        import numpy as np

        probs = self.predict_proba(x_data)
        return np.where(probs >= threshold, 1, 0)

    def score(self, x_data, y_true):
        """计算准确率。"""
        import numpy as np

        y_pred = self.predict(x_data)
        return float(np.mean(y_pred == y_true))


def build_xor_like_data(n_per_group=60, noise=0.15, seed=42):
    """
    构造 XOR 风格二分类数据（非线性可分）。
    - (0,0) 和 (1,1) 标为 0
    - (0,1) 和 (1,0) 标为 1
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    centers = [
        ([0.0, 0.0], 0),
        ([1.0, 1.0], 0),
        ([0.0, 1.0], 1),
        ([1.0, 0.0], 1),
    ]

    x_list = []
    y_list = []
    for center, label in centers:
        samples = rng.normal(loc=center, scale=noise, size=(n_per_group, 2))
        x_list.append(samples)
        y_list.append(np.full(n_per_group, label))

    x_data = np.vstack(x_list)
    y_data = np.concatenate(y_list)

    # 打乱并划分训练/测试。
    indices = rng.permutation(len(x_data))
    x_data = x_data[indices]
    y_data = y_data[indices]
    split = int(0.8 * len(x_data))
    return x_data[:split], x_data[split:], y_data[:split], y_data[split:]


def plot_result(model, x_train, y_train):
    """绘制决策区域和损失曲线。"""
    import numpy as np

    plt = _get_styled_pyplot()
    if plt is None:
        print("未安装 matplotlib，跳过绘图。可安装：pip install matplotlib")
        return

    plt.figure(figsize=(10, 4.5))

    # 子图1：决策区域 + 训练样本。
    plt.subplot(1, 2, 1)
    x_min, x_max = x_train[:, 0].min() - 0.3, x_train[:, 0].max() + 0.3
    y_min, y_max = x_train[:, 1].min() - 0.3, x_train[:, 1].max() + 0.3
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 240),
        np.linspace(y_min, y_max, 240),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    zz = model.predict(grid).reshape(xx.shape)
    plt.contourf(xx, yy, zz, alpha=0.2, cmap="tab10")
    plt.scatter(
        x_train[:, 0],
        x_train[:, 1],
        c=y_train,
        cmap="tab10",
        s=40,
        edgecolors="black",
        alpha=0.85,
    )
    plt.title("两层 MLP 决策区域")
    plt.xlabel("特征 1")
    plt.ylabel("特征 2")

    # 子图2：损失曲线。
    plt.subplot(1, 2, 2)
    plt.plot(model.loss_history, color="tab:green")
    plt.title("训练损失曲线（BCE）")
    plt.xlabel("epoch")
    plt.ylabel("loss")

    plt.tight_layout()
    plt.show()


def main():
    # 1. 检查 numpy 依赖。
    try:
        import numpy  # noqa: F401
    except ImportError:
        print("未安装 numpy。请先执行：pip install numpy")
        return

    # 2. 准备 XOR 风格数据。
    x_train, x_test, y_train, y_test = build_xor_like_data(
        n_per_group=60, noise=0.15, seed=42
    )

    # 3. 创建并训练两层神经网络。
    model = TwoLayerMLPScratch(
        input_size=2,
        hidden_size=8,
        learning_rate=0.2,
        epochs=4000,
        random_state=42,
    )
    model.fit(x_train, y_train, print_every=500)

    # 4. 评估结果。
    train_acc = model.score(x_train, y_train)
    test_acc = model.score(x_test, y_test)
    print("=== MLP（手写两层教学版）===")
    print(f"训练集准确率：{train_acc:.4f}")
    print(f"测试集准确率：{test_acc:.4f}")

    # 5. 绘图展示。
    plot_result(model, x_train, y_train)


if __name__ == "__main__":
    main()
