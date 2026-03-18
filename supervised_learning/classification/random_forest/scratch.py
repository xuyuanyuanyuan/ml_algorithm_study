"""随机森林（手写教学简化版）: Bagging + 决策树桩投票。"""


class DecisionStump:
    """仅做一次划分的树桩。"""

    def __init__(self):
        self.feature_idx = 0
        self.threshold = 0.0
        self.left_label = 0
        self.right_label = 1

    def fit(self, x, y, feature_indices):
        import numpy as np

        best_acc = -1.0
        for feat in feature_indices:
            values = np.unique(x[:, feat])
            for thr in values:
                left = y[x[:, feat] <= thr]
                right = y[x[:, feat] > thr]
                if len(left) == 0 or len(right) == 0:
                    continue
                left_label = int(np.bincount(left).argmax())
                right_label = int(np.bincount(right).argmax())
                pred = np.where(x[:, feat] <= thr, left_label, right_label)
                acc = (pred == y).mean()
                if acc > best_acc:
                    best_acc = acc
                    self.feature_idx = int(feat)
                    self.threshold = float(thr)
                    self.left_label = left_label
                    self.right_label = right_label

    def predict(self, x):
        import numpy as np

        return np.where(
            x[:, self.feature_idx] <= self.threshold, self.left_label, self.right_label
        )


class RandomForestScratch:
    """教学版随机森林。"""

    def __init__(self, n_estimators=15, max_features=2, random_state=42):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.random_state = random_state
        self.stumps = []

    def fit(self, x, y):
        import numpy as np

        rng = np.random.default_rng(self.random_state)
        n_samples, n_features = x.shape
        self.stumps = []

        for _ in range(self.n_estimators):
            sample_idx = rng.integers(0, n_samples, size=n_samples)
            feat_idx = rng.choice(n_features, size=self.max_features, replace=False)
            stump = DecisionStump()
            stump.fit(x[sample_idx], y[sample_idx], feat_idx)
            self.stumps.append(stump)

    def predict(self, x):
        import numpy as np

        all_pred = np.column_stack([stump.predict(x) for stump in self.stumps])
        out = []
        for row in all_pred:
            out.append(int(np.bincount(row).argmax()))
        return np.array(out)


def main():
    import numpy as np

    rng = np.random.default_rng(42)
    x0 = rng.normal(loc=[0, 0], scale=[0.9, 0.9], size=(120, 2))
    x1 = rng.normal(loc=[2.5, 2.2], scale=[0.9, 0.9], size=(120, 2))
    x = np.vstack([x0, x1])
    y = np.array([0] * 120 + [1] * 120)

    model = RandomForestScratch(n_estimators=25, max_features=2, random_state=42)
    model.fit(x, y)
    pred = model.predict(x)
    acc = float((pred == y).mean())
    print("=== Random Forest (scratch) ===")
    print(f"训练准确率={acc:.4f}")


if __name__ == "__main__":
    main()
