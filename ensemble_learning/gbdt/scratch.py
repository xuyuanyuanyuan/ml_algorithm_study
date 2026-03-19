"""GBDT（手写教学简化版）: 用“负梯度残差 + 回归树桩”理解提升。"""


class RegressionStump:
    """最小回归树桩：按单特征阈值分成左右两块并输出均值。"""

    def __init__(self):
        self.feature_idx = 0
        self.threshold = 0.0
        self.left_value = 0.0
        self.right_value = 0.0

    def fit(self, x, target):
        import numpy as np

        best_loss = float("inf")
        for feat in range(x.shape[1]):
            thresholds = np.unique(x[:, feat])
            for thr in thresholds:
                left_mask = x[:, feat] <= thr
                right_mask = ~left_mask
                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    continue
                left_value = float(target[left_mask].mean())
                right_value = float(target[right_mask].mean())
                pred = np.where(left_mask, left_value, right_value)
                loss = float(((target - pred) ** 2).mean())
                if loss < best_loss:
                    best_loss = loss
                    self.feature_idx = int(feat)
                    self.threshold = float(thr)
                    self.left_value = left_value
                    self.right_value = right_value

    def predict(self, x):
        import numpy as np

        return np.where(
            x[:, self.feature_idx] <= self.threshold, self.left_value, self.right_value
        )


class GBDTScratch:
    """教学版 GBDT（二分类）。"""

    def __init__(self, n_estimators=30, learning_rate=0.2):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.base_score = 0.0
        self.stumps = []

    def _sigmoid(self, z):
        import numpy as np

        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, x, y):
        import numpy as np

        p = float(np.clip(y.mean(), 1e-6, 1 - 1e-6))
        self.base_score = float(np.log(p / (1 - p)))
        score = np.full(len(y), self.base_score)
        self.stumps = []

        for _ in range(self.n_estimators):
            prob = self._sigmoid(score)
            # 对于 logloss，负梯度可以写成 y - p。
            residual = y - prob
            stump = RegressionStump()
            stump.fit(x, residual)
            update = stump.predict(x)
            score += self.learning_rate * update
            self.stumps.append(stump)

    def predict_proba(self, x):
        import numpy as np

        score = np.full(x.shape[0], self.base_score)
        for stump in self.stumps:
            score += self.learning_rate * stump.predict(x)
        return self._sigmoid(score)

    def predict(self, x):
        import numpy as np

        return np.where(self.predict_proba(x) >= 0.5, 1, 0)


def main():
    import numpy as np

    rng = np.random.default_rng(42)
    x0 = rng.normal(loc=[0.0, 0.2], scale=[1.0, 1.0], size=(140, 2))
    x1 = rng.normal(loc=[2.2, 2.0], scale=[1.0, 1.0], size=(140, 2))
    x = np.vstack([x0, x1])
    y = np.array([0] * 140 + [1] * 140)

    model = GBDTScratch(n_estimators=35, learning_rate=0.25)
    model.fit(x, y)
    pred = model.predict(x)
    acc = float((pred == y).mean())

    print("=== GBDT Scratch ===")
    print(f"训练准确率={acc:.4f}")


if __name__ == "__main__":
    main()
