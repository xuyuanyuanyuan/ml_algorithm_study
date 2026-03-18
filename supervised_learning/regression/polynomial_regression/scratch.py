"""多项式回归（手写简化版）: 用梯度下降训练。"""


class PolynomialRegressionScratch:
    """教学版多项式回归。"""

    def __init__(self, degree=3, learning_rate=0.01, epochs=2000):
        self.degree = degree
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = 0.0

    def _expand(self, x):
        """把一维 x 扩展成 [x, x^2, ..., x^degree]。"""
        import numpy as np

        cols = [x**i for i in range(1, self.degree + 1)]
        return np.column_stack(cols)

    def fit(self, x, y):
        import numpy as np

        x_poly = self._expand(x)
        n_samples, n_features = x_poly.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for _ in range(self.epochs):
            pred = x_poly @ self.weights + self.bias
            error = pred - y
            grad_w = (2 / n_samples) * (x_poly.T @ error)
            grad_b = (2 / n_samples) * error.sum()
            self.weights -= self.learning_rate * grad_w
            self.bias -= self.learning_rate * grad_b

    def predict(self, x):
        x_poly = self._expand(x)
        return x_poly @ self.weights + self.bias


def main():
    # 关键参数: degree 控制曲线复杂度。
    import numpy as np

    rng = np.random.default_rng(42)
    x = np.linspace(-2.5, 2.5, 100)
    y = 0.7 * x**3 - 1.0 * x**2 + 0.4 * x + rng.normal(0, 1.5, len(x))

    model = PolynomialRegressionScratch(degree=3, learning_rate=0.01, epochs=3000)
    model.fit(x, y)
    pred = model.predict(x)
    mse = float(((pred - y) ** 2).mean())

    print("=== Polynomial Regression (scratch) ===")
    print(f"MSE={mse:.4f}")


if __name__ == "__main__":
    main()
