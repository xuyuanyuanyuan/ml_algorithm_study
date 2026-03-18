"""LightGBM（手写原理模拟版）: 直方图分箱 + 提升树桩。"""


class LightGBMLikeScratch:
    """教学版: 演示“先分箱再找分裂”的思想。"""

    def __init__(self, n_estimators=25, learning_rate=0.2, n_bins=16):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.n_bins = n_bins
        self.base_score = 0.0
        self.stumps = []
        self.bin_edges = []

    def _sigmoid(self, z):
        import numpy as np

        return 1.0 / (1.0 + np.exp(-z))

    def _build_bins(self, x):
        import numpy as np

        self.bin_edges = []
        for j in range(x.shape[1]):
            edges = np.quantile(x[:, j], np.linspace(0, 1, self.n_bins + 1))
            self.bin_edges.append(np.unique(edges))

    def _fit_stump_on_bins(self, x, residual):
        import numpy as np

        best, best_loss = None, float("inf")
        for feat in range(x.shape[1]):
            for thr in self.bin_edges[feat]:
                left = residual[x[:, feat] <= thr]
                right = residual[x[:, feat] > thr]
                if len(left) == 0 or len(right) == 0:
                    continue
                lv, rv = float(left.mean()), float(right.mean())
                pred = np.where(x[:, feat] <= thr, lv, rv)
                loss = ((residual - pred) ** 2).mean()
                if loss < best_loss:
                    best_loss, best = loss, (feat, float(thr), lv, rv)
        return best

    def fit(self, x, y):
        import numpy as np

        self._build_bins(x)
        p = np.clip(y.mean(), 1e-6, 1 - 1e-6)
        self.base_score = float(np.log(p / (1 - p)))
        score = np.full(len(y), self.base_score)
        self.stumps = []

        for _ in range(self.n_estimators):
            residual = y - self._sigmoid(score)
            stump = self._fit_stump_on_bins(x, residual)
            if stump is None:
                break
            feat, thr, lv, rv = stump
            score += self.learning_rate * np.where(x[:, feat] <= thr, lv, rv)
            self.stumps.append(stump)

    def predict(self, x):
        import numpy as np

        score = np.full(x.shape[0], self.base_score)
        for feat, thr, lv, rv in self.stumps:
            score += self.learning_rate * np.where(x[:, feat] <= thr, lv, rv)
        return np.where(self._sigmoid(score) >= 0.5, 1, 0)


def main():
    import numpy as np

    rng = np.random.default_rng(42)
    x = rng.normal(size=(260, 4))
    y = (x[:, 0] + 0.7 * x[:, 1] - 0.4 * x[:, 2] > 0).astype(int)

    model = LightGBMLikeScratch(n_estimators=30, learning_rate=0.2, n_bins=12)
    model.fit(x, y)
    pred = model.predict(x)
    acc = float((pred == y).mean())
    print("=== LightGBM-like Scratch ===")
    print(f"训练准确率={acc:.4f}")


if __name__ == "__main__":
    main()
