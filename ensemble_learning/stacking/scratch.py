"""Stacking（手写教学简化版）: 基模型输出作为元模型输入。"""


class ThresholdClassifier:
    """按单特征阈值划分的简单基模型。"""

    def __init__(self):
        self.feature_idx = 0
        self.threshold = 0.0
        self.left_label = 0
        self.right_label = 1

    def fit(self, x, y):
        import numpy as np

        best_acc = -1.0
        for feat in range(x.shape[1]):
            thresholds = np.unique(x[:, feat])
            for thr in thresholds:
                left = y[x[:, feat] <= thr]
                right = y[x[:, feat] > thr]
                if len(left) == 0 or len(right) == 0:
                    continue
                left_label = int(np.bincount(left).argmax())
                right_label = int(np.bincount(right).argmax())
                pred = np.where(x[:, feat] <= thr, left_label, right_label)
                acc = float((pred == y).mean())
                if acc > best_acc:
                    best_acc = acc
                    self.feature_idx = int(feat)
                    self.threshold = float(thr)
                    self.left_label = left_label
                    self.right_label = right_label

    def predict_proba(self, x):
        import numpy as np

        label = np.where(
            x[:, self.feature_idx] <= self.threshold, self.left_label, self.right_label
        )
        # 给元学习器更平滑的输入，避免全是 0/1。
        return np.where(label == 1, 0.85, 0.15)


class CentroidClassifier:
    """按到两类中心的距离做分类的简单基模型。"""

    def __init__(self):
        self.center0 = None
        self.center1 = None

    def fit(self, x, y):
        self.center0 = x[y == 0].mean(axis=0)
        self.center1 = x[y == 1].mean(axis=0)

    def predict_proba(self, x):
        import numpy as np

        d0 = np.linalg.norm(x - self.center0, axis=1)
        d1 = np.linalg.norm(x - self.center1, axis=1)
        score = d0 - d1
        return 1.0 / (1.0 + np.exp(-score))


class LogisticMetaClassifier:
    """用于二层融合的教学版逻辑回归。"""

    def __init__(self, lr=0.1, n_iter=500):
        self.lr = lr
        self.n_iter = n_iter
        self.w = None
        self.b = 0.0

    def _sigmoid(self, z):
        import numpy as np

        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, x, y):
        import numpy as np

        self.w = np.zeros(x.shape[1], dtype=float)
        self.b = 0.0
        for _ in range(self.n_iter):
            z = x @ self.w + self.b
            p = self._sigmoid(z)
            grad_w = (x.T @ (p - y)) / len(y)
            grad_b = float((p - y).mean())
            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b

    def predict_proba(self, x):
        return self._sigmoid(x @ self.w + self.b)


class StackingScratch:
    """教学版 Stacking：两层结构。"""

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.base_models = [ThresholdClassifier(), CentroidClassifier()]
        self.meta_model = LogisticMetaClassifier(lr=0.2, n_iter=600)

    def fit(self, x, y):
        import numpy as np

        rng = np.random.default_rng(self.random_state)
        idx = np.arange(len(y))
        rng.shuffle(idx)
        split = int(len(y) * 0.7)
        base_idx = idx[:split]
        meta_idx = idx[split:]

        x_base, y_base = x[base_idx], y[base_idx]
        x_meta, y_meta = x[meta_idx], y[meta_idx]

        # 第 1 层：训练基模型。
        for model in self.base_models:
            model.fit(x_base, y_base)

        # 第 2 层：基模型输出作为元模型输入。
        meta_features = np.column_stack([m.predict_proba(x_meta) for m in self.base_models])
        self.meta_model.fit(meta_features, y_meta)

        # 预测阶段更稳定：再用全量数据训练一次基模型。
        for model in self.base_models:
            model.fit(x, y)

    def predict_proba(self, x):
        import numpy as np

        meta_features = np.column_stack([m.predict_proba(x) for m in self.base_models])
        return self.meta_model.predict_proba(meta_features)

    def predict(self, x):
        import numpy as np

        return np.where(self.predict_proba(x) >= 0.5, 1, 0)


def main():
    import numpy as np

    rng = np.random.default_rng(42)
    x0 = rng.normal(loc=[0.0, 0.0], scale=[1.0, 1.0], size=(160, 2))
    x1 = rng.normal(loc=[2.1, 2.3], scale=[1.0, 1.0], size=(160, 2))
    x = np.vstack([x0, x1])
    y = np.array([0] * 160 + [1] * 160)

    model = StackingScratch(random_state=42)
    model.fit(x, y)
    pred = model.predict(x)
    acc = float((pred == y).mean())

    print("=== Stacking Scratch ===")
    print(f"训练准确率={acc:.4f}")


if __name__ == "__main__":
    main()
