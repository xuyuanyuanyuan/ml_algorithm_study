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

    def __init__(self, learning_rate=0.01, lambda_param=0.01, epochs=1000):
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
        reg_part = self.lambda_param * np.sum(self.weights ** 2)
        return hinge_part + reg_part

    def fit(self, x_data, y_data, print_every=100):
        """训练模型。"""
        import numpy as np

        x_data = np.asarray(x_data, dtype=float)
        y_binary = self._prepare_labels(y_data)
        n_samples, n_features = x_data.shape

        # 参数初始化
        self.weights = np.zeros(n_features, dtype=float)
        self.bias = 0.0
        self.loss_history = []

        # 训练循环
        for epoch in range(1, self.epochs + 1):
            for index, sample in enumerate(x_data):
                y_i = y_binary[index]
                condition = y_i * (sample @ self.weights + self.bias) >= 1

                # 梯度计算与参数更新
                if condition:
                    grad_w = 2 * self.lambda_param * self.weights
                    grad_b = 0.0
                else:
                    grad_w = 2 * self.lambda_param * self.weights - y_i * sample
                    grad_b = -y_i

                self.weights -= self.learning_rate * grad_w
                self.bias -= self.learning_rate * grad_b

            # 记录损失
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


def load_iris_binary_from_local(test_ratio=0.2, seed=42):
    """
    从项目 data/iris/iris.data 读取鸢尾花数据。
    只保留两类：
    - Iris-setosa -> 0
    - Iris-versicolor -> 1

    并且只取前两个特征，方便二维可视化。
    返回：
    x_train, x_test, y_train, y_test
    """
    import csv
    from pathlib import Path
    import numpy as np

    # 自动寻找项目根目录
    project_root = None
    for parent in Path(__file__).resolve().parents:
        if (parent / "data" / "iris" / "iris.data").exists():
            project_root = parent
            break

    if project_root is None:
        raise FileNotFoundError("未找到 data/iris/iris.data，请检查项目目录结构。")

    iris_path = project_root / "data" / "iris" / "iris.data"

    x_list = []
    y_list = []

    with open(iris_path, "r", encoding="utf-8") as file:
        reader = csv.reader(file)
        for row in reader:
            if not row or len(row) < 5:
                continue

            label_name = row[4].strip()

            # 只保留两类，做二分类
            if label_name == "Iris-setosa":
                label = 0
            elif label_name == "Iris-versicolor":
                label = 1
            else:
                # 丢掉 Iris-virginica，避免变成三分类
                continue

            # 只取前两个特征，方便画二维图
            features = [float(row[0]), float(row[1])]
            x_list.append(features)
            y_list.append(label)

    x_data = np.array(x_list, dtype=float)
    y_data = np.array(y_list, dtype=int)

    # 打乱
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(x_data))
    x_data = x_data[indices]
    y_data = y_data[indices]

    # 划分训练集 / 测试集
    split = int(len(x_data) * (1 - test_ratio))
    x_train = x_data[:split]
    x_test = x_data[split:]
    y_train = y_data[:split]
    y_test = y_data[split:]

    return x_train, x_test, y_train, y_test


def plot_result(model, x_train, y_train, x_test, y_test):
    """可视化样本点、决策边界和训练损失。"""
    import numpy as np

    plt = _get_styled_pyplot()
    if plt is None:
        print("未安装 matplotlib，跳过绘图。可安装：pip install matplotlib")
        return

    plt.figure(figsize=(10, 4.5))

    # 子图1：数据点 + 决策边界
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

    # 背景决策区域
    x_min, x_max = x_train[:, 0].min() - 0.8, x_train[:, 0].max() + 0.8
    y_min, y_max = x_train[:, 1].min() - 0.8, x_train[:, 1].max() + 0.8
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 240),
        np.linspace(y_min, y_max, 240),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    zz = model.predict(grid).reshape(xx.shape)
    plt.contourf(xx, yy, zz, alpha=0.18, cmap="tab10")

    # 决策边界与 margin
    if abs(model.weights[1]) > 1e-12:
        x_line = np.linspace(x_min, x_max, 200)
        y_line = -(model.weights[0] * x_line + model.bias) / model.weights[1]
        y_margin_pos = -(model.weights[0] * x_line + model.bias - 1) / model.weights[1]
        y_margin_neg = -(model.weights[0] * x_line + model.bias + 1) / model.weights[1]
        plt.plot(x_line, y_line, color="black", label="决策边界")
        plt.plot(x_line, y_margin_pos, color="black", linestyle="--", alpha=0.7)
        plt.plot(x_line, y_margin_neg, color="black", linestyle="--", alpha=0.7)

    plt.title("线性 SVM 分类结果（Iris 二分类）")
    plt.xlabel("特征 1（萼片长度）")
    plt.ylabel("特征 2（萼片宽度）")
    plt.legend()

    # 子图2：损失曲线
    plt.subplot(1, 2, 2)
    plt.plot(model.loss_history, color="tab:green")
    plt.title("训练损失曲线（hinge loss）")
    plt.xlabel("epoch")
    plt.ylabel("loss")

    plt.tight_layout()
    plt.show()


def main():
    # 1. 检查 numpy 依赖
    try:
        import numpy  # noqa: F401
    except ImportError:
        print("未安装 numpy。请先执行：pip install numpy")
        return

    # 2. 从本地 iris.data 读取数据
    x_train, x_test, y_train, y_test = load_iris_binary_from_local(test_ratio=0.2, seed=42)

    # 3. 创建并训练简化版 SVM
    model = LinearSVMScratch(learning_rate=0.01, lambda_param=0.01, epochs=800)
    model.fit(x_train, y_train, print_every=200)

    # 4. 评估结果
    train_acc = model.score(x_train, y_train)
    test_acc = model.score(x_test, y_test)
    print("=== SVM（手写教学简化版，使用本地 Iris 数据）===")
    print(f"训练集准确率：{train_acc:.4f}")
    print(f"测试集准确率：{test_acc:.4f}")
    print(f"学习到的权重：{model.weights}")
    print(f"学习到的偏置：{model.bias:.4f}")

    # 5. 绘图展示
    plot_result(model, x_train, y_train, x_test, y_test)


if __name__ == "__main__":
    main()