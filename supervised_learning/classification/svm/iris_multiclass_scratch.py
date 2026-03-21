"""Iris 三分类 SVM（手写 OVR + 线性 SVM + SGD）。"""


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


def find_project_root():
    """自动寻找项目根目录，优先保证可以读取 data/iris/iris.data。"""
    from pathlib import Path

    for parent in Path(__file__).resolve().parents:
        if (parent / "data" / "iris" / "iris.data").exists():
            return parent
    raise FileNotFoundError("未找到项目根目录下的 data/iris/iris.data")


def load_local_iris_data():
    """
    从项目根目录下读取 iris.data。

    这里为了便于做二维可视化，只取前两个特征：
    - 萼片长度
    - 萼片宽度
    """
    import csv
    import numpy as np

    project_root = find_project_root()
    iris_path = project_root / "data" / "iris" / "iris.data"

    label_to_id = {
        "Iris-setosa": 0,
        "Iris-versicolor": 1,
        "Iris-virginica": 2,
    }
    label_names = [
        "Iris-setosa",
        "Iris-versicolor",
        "Iris-virginica",
    ]
    feature_names = [
        "萼片长度",
        "萼片宽度",
    ]

    x_list = []
    y_list = []
    with open(iris_path, "r", encoding="utf-8") as file:
        reader = csv.reader(file)
        for row in reader:
            if not row or len(row) < 5:
                continue

            label_name = row[4].strip()
            if label_name not in label_to_id:
                continue

            # 只取前两个特征，方便画三分类决策区域。
            x_list.append([float(row[0]), float(row[1])])
            y_list.append(label_to_id[label_name])

    x_data = np.asarray(x_list, dtype=float)
    y_data = np.asarray(y_list, dtype=int)
    return x_data, y_data, label_names, feature_names, iris_path


def stratified_train_test_split(x_data, y_data, test_size=0.2, seed=42):
    """手写一个最小分层划分，保证每一类在训练集和测试集中都有样本。"""
    import numpy as np

    rng = np.random.default_rng(seed)
    train_indices = []
    test_indices = []

    for class_id in np.unique(y_data):
        class_indices = np.where(y_data == class_id)[0]
        rng.shuffle(class_indices)

        n_test = int(round(len(class_indices) * test_size))
        n_test = max(1, min(n_test, len(class_indices) - 1))

        test_indices.extend(class_indices[:n_test])
        train_indices.extend(class_indices[n_test:])

    train_indices = np.asarray(train_indices, dtype=int)
    test_indices = np.asarray(test_indices, dtype=int)
    rng.shuffle(train_indices)
    rng.shuffle(test_indices)

    return (
        x_data[train_indices],
        x_data[test_indices],
        y_data[train_indices],
        y_data[test_indices],
    )


class StandardScalerScratch:
    """最小标准化工具，让手写 SVM 更容易训练。"""

    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, x_data):
        import numpy as np

        x_data = np.asarray(x_data, dtype=float)
        self.mean_ = x_data.mean(axis=0)
        self.scale_ = x_data.std(axis=0)
        self.scale_ = np.where(self.scale_ < 1e-12, 1.0, self.scale_)
        return self

    def transform(self, x_data):
        import numpy as np

        x_data = np.asarray(x_data, dtype=float)
        return (x_data - self.mean_) / self.scale_

    def fit_transform(self, x_data):
        self.fit(x_data)
        return self.transform(x_data)


class LinearBinarySVMScratch:
    """
    手写线性二分类 SVM。

    说明：
    - 使用 hinge loss（合页损失）
    - 使用 L2 正则项
    - 使用随机梯度下降 SGD 训练
    - 输出的是一个二分类超平面
    """

    def __init__(self, learning_rate=0.01, lambda_param=0.01, epochs=1000, random_state=42):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.epochs = epochs
        self.random_state = random_state
        self.weights = None
        self.bias = 0.0
        self.loss_history = []

    def _prepare_labels(self, y_data):
        """把标签转成二分类 SVM 常用的 -1 / +1。"""
        import numpy as np

        y_data = np.asarray(y_data)
        return np.where(y_data > 0, 1.0, -1.0)

    def _hinge_loss(self, x_data, y_binary):
        """计算平均 hinge loss + L2 正则项。"""
        import numpy as np

        scores = x_data @ self.weights + self.bias
        margins = 1.0 - y_binary * scores
        hinge_part = np.maximum(0.0, margins).mean()
        reg_part = self.lambda_param * float(np.sum(self.weights**2))
        return hinge_part + reg_part

    def fit(self, x_data, y_data, print_every=0):
        """训练手写二分类 SVM。"""
        import numpy as np

        rng = np.random.default_rng(self.random_state)
        x_data = np.asarray(x_data, dtype=float)
        y_binary = self._prepare_labels(y_data)
        n_samples, n_features = x_data.shape

        self.weights = np.zeros(n_features, dtype=float)
        self.bias = 0.0
        self.loss_history = []

        for epoch in range(1, self.epochs + 1):
            indices = rng.permutation(n_samples)

            for index in indices:
                sample = x_data[index]
                y_i = y_binary[index]
                margin = y_i * (sample @ self.weights + self.bias)

                if margin >= 1.0:
                    grad_w = 2.0 * self.lambda_param * self.weights
                    grad_b = 0.0
                else:
                    grad_w = 2.0 * self.lambda_param * self.weights - y_i * sample
                    grad_b = -y_i

                self.weights -= self.learning_rate * grad_w
                self.bias -= self.learning_rate * grad_b

            loss = self._hinge_loss(x_data, y_binary)
            self.loss_history.append(loss)
            if print_every and epoch % print_every == 0:
                print(f"第 {epoch:4d} 轮，loss={loss:.6f}")

        return self

    def decision_function(self, x_data):
        """返回样本到超平面的距离分数。"""
        import numpy as np

        x_data = np.asarray(x_data, dtype=float)
        return x_data @ self.weights + self.bias

    def predict(self, x_data):
        """二分类预测：分数 >= 0 记为正类。"""
        import numpy as np

        scores = self.decision_function(x_data)
        return np.where(scores >= 0, 1, 0)


class OVRMulticlassSVM:
    """
    One-vs-Rest（OVR）多分类 SVM。

    为什么二分类 SVM 不能直接做三分类？
    - 经典线性 SVM 的输出本质上是一条分界线（或高维超平面）。
    - 它天然回答的是“样本在分界线哪一边”，适合两类问题。
    - 当类别数变成 3 类时，一个单独的二分类超平面不够直接给出三分类结果。

    为什么这里选择 OVR？
    - OVR 是最容易理解的多分类改造方法之一。
    - 对每个类别训练一个“该类 vs 其余所有类”的二分类器。
    - 预测时，把样本交给所有分类器打分，谁的分数最高，就认为它最像哪一类。

    三分类 Iris 的训练方式：
    - 分类器 1：Iris-setosa 作为正类，其余两类作为负类
    - 分类器 2：Iris-versicolor 作为正类，其余两类作为负类
    - 分类器 3：Iris-virginica 作为正类，其余两类作为负类
    """

    def __init__(self, learning_rate=0.01, lambda_param=0.01, epochs=1000, random_state=42):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.epochs = epochs
        self.random_state = random_state
        self.scaler = StandardScalerScratch()
        self.classes_ = None
        self.label_names_ = None
        self.classifiers = {}

    def fit(self, x_data, y_data, label_names=None):
        import numpy as np

        x_data = np.asarray(x_data, dtype=float)
        y_data = np.asarray(y_data, dtype=int)

        self.classes_ = np.unique(y_data)
        self.label_names_ = label_names if label_names is not None else [str(i) for i in self.classes_]

        # 先对特征做标准化，避免某个维度尺度过大，影响 SGD 训练稳定性。
        x_scaled = self.scaler.fit_transform(x_data)
        self.classifiers = {}

        for class_id in self.classes_:
            # 当前类别是正类，其余所有类别合成负类。
            y_binary = np.where(y_data == class_id, 1, 0)
            classifier = LinearBinarySVMScratch(
                learning_rate=self.learning_rate,
                lambda_param=self.lambda_param,
                epochs=self.epochs,
                random_state=self.random_state + int(class_id),
            )
            classifier.fit(x_scaled, y_binary)
            self.classifiers[int(class_id)] = classifier

        return self

    def decision_function(self, x_data):
        """返回每个样本在每个 OVR 分类器上的分数。"""
        import numpy as np

        x_scaled = self.scaler.transform(x_data)
        score_list = []
        for class_id in self.classes_:
            classifier = self.classifiers[int(class_id)]
            score_list.append(classifier.decision_function(x_scaled))
        return np.column_stack(score_list)

    def predict(self, x_data):
        """分数最高的那个分类器，对应最终类别。"""
        scores = self.decision_function(x_data)
        best_indices = scores.argmax(axis=1)
        return self.classes_[best_indices]

    def score(self, x_data, y_true):
        import numpy as np

        y_true = np.asarray(y_true, dtype=int)
        y_pred = self.predict(x_data)
        return float((y_true == y_pred).mean())

    def get_original_space_params(self, class_id):
        """
        把标准化空间中的参数换回原始特征空间，方便初学者理解。

        如果标准化后模型是：
            score = w * ((x - mean) / std) + b
        那么原始空间可写成：
            score = w' * x + b'
        """
        import numpy as np

        classifier = self.classifiers[int(class_id)]
        original_weights = classifier.weights / self.scaler.scale_
        original_bias = classifier.bias - float(
            np.sum(classifier.weights * self.scaler.mean_ / self.scaler.scale_)
        )
        return original_weights, original_bias


def plot_multiclass_result(model, x_train, y_train, x_test, y_test, label_names, feature_names):
    """展示三类散点图、三分类决策区域，以及 OVR 每个分类器的损失曲线。"""
    import numpy as np

    plt = _get_styled_pyplot()
    if plt is None:
        print("未安装 matplotlib，跳过绘图。可安装：pip install matplotlib")
        return

    plt.figure(figsize=(11, 4.8))

    # 子图 1：三分类散点图 + 决策区域 / 决策边界。
    plt.subplot(1, 2, 1)
    x_all = np.vstack([x_train, x_test])
    x_min, x_max = x_all[:, 0].min() - 0.6, x_all[:, 0].max() + 0.6
    y_min, y_max = x_all[:, 1].min() - 0.6, x_all[:, 1].max() + 0.6

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 320),
        np.linspace(y_min, y_max, 320),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    zz = model.predict(grid).reshape(xx.shape)

    plt.contourf(xx, yy, zz, alpha=0.18, cmap="tab10")
    plt.contour(xx, yy, zz, levels=[0.5, 1.5], colors="black", linewidths=1.0, alpha=0.65)

    for class_id, class_name in enumerate(label_names):
        train_mask = y_train == class_id
        test_mask = y_test == class_id

        plt.scatter(
            x_train[train_mask, 0],
            x_train[train_mask, 1],
            s=42,
            alpha=0.85,
            c=np.full(train_mask.sum(), class_id),
            cmap="tab10",
            label=f"训练集 - {class_name}",
        )
        plt.scatter(
            x_test[test_mask, 0],
            x_test[test_mask, 1],
            s=75,
            marker="x",
            linewidths=1.8,
            c=np.full(test_mask.sum(), class_id),
            cmap="tab10",
            label=f"测试集 - {class_name}",
        )

    plt.title("Iris 三分类决策区域（手写 OVR-SVM）")
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    plt.legend(fontsize=8)

    # 子图 2：每个 OVR 二分类器的 loss 曲线。
    plt.subplot(1, 2, 2)
    for class_id, class_name in enumerate(label_names):
        classifier = model.classifiers[class_id]
        plt.plot(classifier.loss_history, label=f"{class_name} vs Rest")

    plt.title("三个 OVR 分类器的训练损失")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.show()


def main():
    # 依赖说明：
    # - numpy：数值计算
    # - matplotlib：可视化（可选）
    try:
        import numpy as np  # noqa: F401
    except ImportError:
        print("未安装 numpy。请先执行：pip install numpy")
        return

    # 1. 从项目根目录读取 iris.data，只取前两个特征。
    try:
        x_data, y_data, label_names, feature_names, iris_path = load_local_iris_data()
    except FileNotFoundError as exc:
        print(exc)
        return

    # 2. 划分训练集 / 测试集。
    x_train, x_test, y_train, y_test = stratified_train_test_split(
        x_data,
        y_data,
        test_size=0.2,
        seed=42,
    )

    # 3. 训练手写三分类 SVM。
    model = OVRMulticlassSVM(
        learning_rate=0.01,
        lambda_param=0.01,
        epochs=1000,
        random_state=42,
    )
    model.fit(x_train, y_train, label_names=label_names)

    # 4. 输出准确率。
    train_acc = model.score(x_train, y_train)
    test_acc = model.score(x_test, y_test)

    print("=== Iris 三分类 SVM（手写 OVR + 线性 SVM）===")
    print(f"数据文件：{iris_path}")
    print(f"训练集样本数：{len(x_train)}")
    print(f"测试集样本数：{len(x_test)}")
    print(f"训练集准确率：{train_acc:.4f}")
    print(f"测试集准确率：{test_acc:.4f}")
    print("说明：下面输出的是换回原始特征空间后的权重和偏置，更方便理解决策边界。")

    # 5. 输出每个 OVR 分类器的权重和偏置。
    for class_id, class_name in enumerate(label_names):
        weights, bias = model.get_original_space_params(class_id)
        classifier = model.classifiers[class_id]
        print(f"\n[{class_name} vs Rest]")
        print(f"权重：{weights}")
        print(f"偏置：{bias:.6f}")
        print(f"最后一轮 loss：{classifier.loss_history[-1]:.6f}")

    # 6. 可视化展示。
    plot_multiclass_result(
        model,
        x_train,
        y_train,
        x_test,
        y_test,
        label_names,
        feature_names,
    )


if __name__ == "__main__":
    main()
