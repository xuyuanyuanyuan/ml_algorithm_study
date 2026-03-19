"""AdaBoost（手写教学简化版）: 以加权样本训练弱分类器并线性加权。"""


class WeightedDecisionStump:
    """带样本权重的一层树桩。"""

    def __init__(self):
        self.feature_idx = 0
        self.threshold = 0.0
        self.polarity = 1

    def fit(self, x, y_pm, sample_weight):
        import numpy as np

        best_error = float("inf")
        for feat in range(x.shape[1]):
            thresholds = np.unique(x[:, feat])
            for thr in thresholds:
                for polarity in (1, -1):
                    pred = np.ones(len(y_pm))
                    pred[x[:, feat] <= thr] = -1
                    pred = pred * polarity
                    error = float(sample_weight[pred != y_pm].sum())
                    if error < best_error:
                        best_error = error
                        self.feature_idx = int(feat)
                        self.threshold = float(thr)
                        self.polarity = int(polarity)

    def predict(self, x):
        import numpy as np

        pred = np.ones(x.shape[0])
        pred[x[:, self.feature_idx] <= self.threshold] = -1
        return pred * self.polarity


class AdaBoostScratch:
    """教学版 AdaBoost（二分类）。"""

    def __init__(self, n_estimators=30):
        self.n_estimators = n_estimators
        self.stumps = []
        self.alphas = []

    def fit(self, x, y):
        import numpy as np

        y_pm = np.where(y > 0, 1.0, -1.0)
        w = np.full(len(y), 1.0 / len(y))
        self.stumps = []
        self.alphas = []

        for _ in range(self.n_estimators):
            stump = WeightedDecisionStump()
            stump.fit(x, y_pm, w)
            pred = stump.predict(x)
            err = float(np.clip(w[pred != y_pm].sum(), 1e-10, 1 - 1e-10))
            alpha = 0.5 * float(np.log((1 - err) / err))

            # 错分样本权重会被放大，促使后续弱分类器关注难样本。
            w *= np.exp(-alpha * y_pm * pred)
            w /= w.sum()

            self.stumps.append(stump)
            self.alphas.append(alpha)

    def predict(self, x):
        import numpy as np

        score = np.zeros(x.shape[0])
        for alpha, stump in zip(self.alphas, self.stumps):
            score += alpha * stump.predict(x)
        return np.where(score >= 0, 1, 0)


def main():
    import numpy as np

    rng = np.random.default_rng(42)
    x0 = rng.normal(loc=[0.0, 0.0], scale=[1.0, 1.0], size=(130, 2))
    x1 = rng.normal(loc=[2.0, 2.0], scale=[1.0, 1.0], size=(130, 2))
    x = np.vstack([x0, x1])
    y = np.array([0] * 130 + [1] * 130)

    model = AdaBoostScratch(n_estimators=40)
    model.fit(x, y)
    pred = model.predict(x)
    acc = float((pred == y).mean())

    print("=== AdaBoost Scratch ===")
    print(f"训练准确率={acc:.4f}")


if __name__ == "__main__":
    main()
