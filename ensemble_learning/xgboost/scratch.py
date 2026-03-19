"""XGBoost（手写教学原理版）: 用梯度/二阶梯度与增益做最小示例。"""


class XGBoostStump:
    """教学树桩：用一阶梯度和二阶梯度寻找最佳划分。"""

    def __init__(self, reg_lambda=1.0, gamma=0.0):
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.has_split = False
        self.feature_idx = 0
        self.threshold = 0.0
        self.left_weight = 0.0
        self.right_weight = 0.0

    def fit(self, x, grad, hess):
        import numpy as np

        total_grad = float(grad.sum())
        total_hess = float(hess.sum())
        base_term = (total_grad ** 2) / (total_hess + self.reg_lambda)

        best_gain = -float("inf")
        best_split = None
        for feat in range(x.shape[1]):
            thresholds = np.unique(x[:, feat])
            for thr in thresholds:
                left_mask = x[:, feat] <= thr
                right_mask = ~left_mask
                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    continue

                g_l = float(grad[left_mask].sum())
                h_l = float(hess[left_mask].sum())
                g_r = float(grad[right_mask].sum())
                h_r = float(hess[right_mask].sum())

                gain = 0.5 * (
                    (g_l ** 2) / (h_l + self.reg_lambda)
                    + (g_r ** 2) / (h_r + self.reg_lambda)
                    - base_term
                ) - self.gamma

                if gain > best_gain:
                    best_gain = gain
                    left_weight = -g_l / (h_l + self.reg_lambda)
                    right_weight = -g_r / (h_r + self.reg_lambda)
                    best_split = (int(feat), float(thr), float(left_weight), float(right_weight))

        if best_split is None or best_gain <= 0:
            # 若没有有效划分，退化成常数叶节点。
            weight = -total_grad / (total_hess + self.reg_lambda)
            self.has_split = False
            self.left_weight = float(weight)
            self.right_weight = float(weight)
            return

        self.has_split = True
        self.feature_idx, self.threshold, self.left_weight, self.right_weight = best_split

    def predict(self, x):
        import numpy as np

        if not self.has_split:
            return np.full(x.shape[0], self.left_weight)
        return np.where(
            x[:, self.feature_idx] <= self.threshold, self.left_weight, self.right_weight
        )


class XGBoostScratch:
    """教学版 XGBoost（二分类）。"""

    def __init__(self, n_estimators=40, learning_rate=0.2, reg_lambda=1.0, gamma=0.0):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.reg_lambda = reg_lambda
        self.gamma = gamma
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
            grad = prob - y
            hess = np.clip(prob * (1 - prob), 1e-6, None)

            stump = XGBoostStump(reg_lambda=self.reg_lambda, gamma=self.gamma)
            stump.fit(x, grad, hess)
            score += self.learning_rate * stump.predict(x)
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
    x0 = rng.normal(loc=[0.0, 0.0], scale=[1.0, 1.0], size=(150, 2))
    x1 = rng.normal(loc=[2.4, 2.1], scale=[1.0, 1.0], size=(150, 2))
    x = np.vstack([x0, x1])
    y = np.array([0] * 150 + [1] * 150)

    model = XGBoostScratch(n_estimators=40, learning_rate=0.25, reg_lambda=1.0, gamma=0.0)
    model.fit(x, y)
    pred = model.predict(x)
    acc = float((pred == y).mean())

    print("=== XGBoost Scratch ===")
    print(f"训练准确率={acc:.4f}, 树桩数量={len(model.stumps)}")


if __name__ == "__main__":
    main()
