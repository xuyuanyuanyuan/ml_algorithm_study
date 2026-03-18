"""Lasso 回归（手写简化版）: 坐标下降 + 软阈值。"""


class LassoScratch:
    """教学版 Lasso (L1 正则线性回归)。"""

    def __init__(self, alpha=0.1, epochs=200):
        self.alpha = alpha
        self.epochs = epochs
        self.coef_ = None
        self.intercept_ = 0.0

    def _soft_threshold(self, value, lam):
        if value > lam:
            return value - lam
        if value < -lam:
            return value + lam
        return 0.0

    def fit(self, x, y):
        import numpy as np

        n_samples, n_features = x.shape
        self.coef_ = np.zeros(n_features)
        self.intercept_ = y.mean()
        y_center = y - self.intercept_

        for _ in range(self.epochs):
            for j in range(n_features):
                # 计算去掉第 j 个特征后的残差。
                y_pred = x @ self.coef_
                residual = y_center - y_pred + x[:, j] * self.coef_[j]
                rho = (x[:, j] * residual).sum()
                z = (x[:, j] ** 2).sum()
                self.coef_[j] = self._soft_threshold(rho, self.alpha * n_samples) / (z + 1e-12)

    def predict(self, x):
        return x @ self.coef_ + self.intercept_


def main():
    # 关键参数: alpha 控制稀疏程度。
    import numpy as np

    rng = np.random.default_rng(42)
    x = rng.normal(size=(180, 12))
    true_w = np.array([3.0, 0.0, -2.0, 0.0, 1.5, 0.0, 0.0, 2.2, 0.0, 0.0, 0.7, 0.0])
    y = x @ true_w + rng.normal(0, 0.8, size=180)

    model = LassoScratch(alpha=0.08, epochs=300)
    model.fit(x, y)
    pred = model.predict(x)
    mse = float(((pred - y) ** 2).mean())
    non_zero = int((abs(model.coef_) > 1e-6).sum())

    print("=== Lasso Regression (scratch) ===")
    print(f"MSE={mse:.4f}, 非零系数个数={non_zero}")


if __name__ == "__main__":
    main()
