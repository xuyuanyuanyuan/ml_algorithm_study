"""XGBoost（手写原理模拟版）: 用“残差 + 树桩”理解提升思想。"""


class XGBoostLikeScratch:
    """教学版: 不是完整 XGBoost, 仅演示 boosting 主流程。"""

    def __init__(self, n_estimators=30, learning_rate=0.2):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.stumps = []  # 每个 stump: (feature, threshold, left_value, right_value)
        self.base_score = 0.0

    def _sigmoid(self, z):
        import numpy as np

        return 1.0 / (1.0 + np.exp(-z))

    def _fit_stump(self, x, residual):
        import numpy as np

        best = None
        best_loss = float("inf")
        for feat in range(x.shape[1]):
            thresholds = np.unique(x[:, feat])
            for thr in thresholds:
                left = residual[x[:, feat] <= thr]
                right = residual[x[:, feat] > thr]
                if len(left) == 0 or len(right) == 0:
                    continue
                lv = left.mean()
                rv = right.mean()
                pred = np.where(x[:, feat] <= thr, lv, rv)
                loss = ((residual - pred) ** 2).mean()
                if loss < best_loss:
                    best_loss = loss
                    best = (feat, float(thr), float(lv), float(rv))
        return best

    def fit(self, x, y):
        import numpy as np

        # 初始分数用对数几率。
        p = np.clip(y.mean(), 1e-6, 1 - 1e-6)
        self.base_score = float(np.log(p / (1 - p)))
        score = np.full(len(y), self.base_score)

        for _ in range(self.n_estimators):
            prob = self._sigmoid(score)
            residual = y - prob
            stump = self._fit_stump(x, residual)
            if stump is None:
                break
            feat, thr, lv, rv = stump
            update = np.where(x[:, feat] <= thr, lv, rv)
            score += self.learning_rate * update
            self.stumps.append(stump)

    def predict_proba(self, x):
        import numpy as np

        score = np.full(x.shape[0], self.base_score)
        for feat, thr, lv, rv in self.stumps:
            score += self.learning_rate * np.where(x[:, feat] <= thr, lv, rv)
        return self._sigmoid(score)

    def predict(self, x):
        import numpy as np

        return np.where(self.predict_proba(x) >= 0.5, 1, 0)


def main():
    import numpy as np

    rng = np.random.default_rng(42)
    x0 = rng.normal(loc=[0, 0], scale=[1, 1], size=(140, 2))
    x1 = rng.normal(loc=[2.3, 2.0], scale=[1, 1], size=(140, 2))
    x = np.vstack([x0, x1])
    y = np.array([0] * 140 + [1] * 140)

    model = XGBoostLikeScratch(n_estimators=35, learning_rate=0.25)
    model.fit(x, y)
    pred = model.predict(x)
    acc = float((pred == y).mean())
    print("=== XGBoost-like Scratch ===")
    print(f"训练准确率={acc:.4f}, 树桩数量={len(model.stumps)}")


if __name__ == "__main__":
    main()
