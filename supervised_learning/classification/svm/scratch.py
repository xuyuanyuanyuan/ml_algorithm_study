"""SVM（手写教学简化版）：使用线性模型 + hinge loss 的 SGD 训练。"""


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


class LinearSVMScratch:
    """
    线性 SVM 的教学型简化实现。
    说明：
    - 只做二分类
    - 使用 hinge loss + L2 正则
    - 使用随机梯度下降（SGD）更新参数
    """

    def __init__(self, learning_rate=0.01, lambda_param=0.0000000001, epochs=1000):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.epochs = epochs
        self.weights = None
        self.bias = 0.0
        self.loss_history = []

    def _prepare_labels(self, y):
        """把标签转为 SVM 常用的 -1 / +1。"""
        import numpy as np

        y = np.asarray(y)
        return np.where(y <= 0, -1.0, 1.0)

    def _hinge_loss(self, x_data, y_binary):
        """计算当前参数下的平均 hinge loss。"""
        import numpy as np

        scores = x_data @ self.weights + self.bias
        margins = 1 - y_binary * scores
        hinge_part = np.maximum(0, margins).mean()
        reg_part = self.lambda_param * np.sum(self.weights**2)
        return hinge_part + reg_part

    def fit(self, x_data, y_data, print_every=100):
        """训练模型。"""
        import numpy as np

        x_data = np.asarray(x_data, dtype=float)
        y_binary = self._prepare_labels(y_data)
        n_samples, n_features = x_data.shape

        # 参数初始化。
        self.weights = np.zeros(n_features, dtype=float)
        self.bias = 0.0
        self.loss_history = []

        # 训练循环。
        for epoch in range(1, self.epochs + 1):
            for index, sample in enumerate(x_data):
                y_i = y_binary[index]
                condition = y_i * (sample @ self.weights + self.bias) >= 1

                # 梯度计算与参数更新。
                if condition:
                    grad_w = 2 * self.lambda_param * self.weights
                    grad_b = 0.0
                else:
                    grad_w = 2 * self.lambda_param * self.weights - y_i * sample
                    grad_b = -y_i

                self.weights -= self.learning_rate * grad_w
                self.bias -= self.learning_rate * grad_b

            # 记录损失，便于观察收敛。
            loss = self._hinge_loss(x_data, y_binary)
            self.loss_history.append(loss)
            if epoch % print_every == 0:
                print(f"第 {epoch:4d} 轮，loss={loss:.6f}")

    def decision_function(self, x_data):
        """返回决策分数。"""
        import numpy as np

        x_data = np.asarray(x_data, dtype=float)
        return x_data @ self.weights + self.bias

    def predict(self, x_data):
        """根据分数正负进行分类。"""
        import numpy as np

        scores = self.decision_function(x_data)
        return np.where(scores >= 0, 1, 0)

    def score(self, x_data, y_true):
        """计算准确率。"""
        import numpy as np

        y_true = np.asarray(y_true)
        y_pred = self.predict(x_data)
        return float((y_true == y_pred).mean())


def build_demo_data(n_per_class=50, seed=42):
    """构造一个近似线性可分的二维二分类数据集。"""
    import numpy as np

    rng = np.random.default_rng(seed)

    class_0 = rng.normal(loc=[1.5, 1.5], scale=[0.55, 0.55], size=(n_per_class, 2))
    class_1 = rng.normal(loc=[4.5, 4.0], scale=[0.60, 0.60], size=(n_per_class, 2))

    x_data = np.vstack([class_0, class_1])
    y_data = np.array([0] * n_per_class + [1] * n_per_class)

    # 打乱并按 8:2 划分训练/测试。
    indices = rng.permutation(len(x_data))
    x_data = x_data[indices]
    y_data = y_data[indices]
    split = int(len(x_data) * 0.8)

    return x_data[:split], x_data[split:], y_data[:split], y_data[split:]


def plot_result(model, x_train, y_train, x_test, y_test):
    """可视化样本点、决策边界和训练损失。"""
    import numpy as np

    plt = _get_styled_pyplot()
    if plt is None:
        print("未安装 matplotlib，跳过绘图。可安装：pip install matplotlib")
        return

    plt.figure(figsize=(10, 4.5))

    # 子图1：数据点 + 决策边界。
    plt.subplot(1, 2, 1)
    plt.scatter(
        x_train[:, 0],
        x_train[:, 1],
        c=y_train,
        cmap="tab10",
        s=45,
        alpha=0.8,
        label="训练集",
    )
    plt.scatter(
        x_test[:, 0],
        x_test[:, 1],
        c=y_test,
        cmap="tab10",
        s=80,
        marker="x",
        linewidths=2,
        label="测试集",
    )

    # 画背景决策区域。
    x_min, x_max = x_train[:, 0].min() - 0.8, x_train[:, 0].max() + 0.8
    y_min, y_max = x_train[:, 1].min() - 0.8, x_train[:, 1].max() + 0.8
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 240),
        np.linspace(y_min, y_max, 240),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    zz = model.predict(grid).reshape(xx.shape)
    plt.contourf(xx, yy, zz, alpha=0.18, cmap="tab10")

    # 画决策边界与两条 margin 线。
    if abs(model.weights[1]) > 1e-12:
        x_line = np.linspace(x_min, x_max, 200)
        y_line = -(model.weights[0] * x_line + model.bias) / model.weights[1]
        y_margin_pos = -(model.weights[0] * x_line + model.bias - 1) / model.weights[1]
        y_margin_neg = -(model.weights[0] * x_line + model.bias + 1) / model.weights[1]
        plt.plot(x_line, y_line, color="black", label="决策边界")
        plt.plot(x_line, y_margin_pos, color="black", linestyle="--", alpha=0.7)
        plt.plot(x_line, y_margin_neg, color="black", linestyle="--", alpha=0.7)

    plt.title("线性 SVM 分类结果")
    plt.xlabel("特征 1")
    plt.ylabel("特征 2")
    plt.legend()

    # 子图2：损失曲线。
    plt.subplot(1, 2, 2)
    plt.plot(model.loss_history, color="tab:green")
    plt.title("训练损失曲线（hinge loss）")
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

    # 2. 构造数据。
    x_train, x_test, y_train, y_test = build_demo_data(n_per_class=50, seed=42)

    # 3. 创建并训练简化版 SVM。
    model = LinearSVMScratch(learning_rate=0.01, lambda_param=0.01, epochs=800)
    model.fit(x_train, y_train, print_every=200)

    # 4. 评估结果。
    train_acc = model.score(x_train, y_train)
    test_acc = model.score(x_test, y_test)
    print("=== SVM（手写教学简化版）===")
    print(f"训练集准确率：{train_acc:.4f}")
    print(f"测试集准确率：{test_acc:.4f}")
    print(f"学习到的权重：{model.weights}")
    print(f"学习到的偏置：{model.bias:.4f}")

    # 5. 绘图展示。
    plot_result(model, x_train, y_train, x_test, y_test)


if __name__ == "__main__":
    main()
