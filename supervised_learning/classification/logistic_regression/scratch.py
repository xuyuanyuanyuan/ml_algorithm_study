"""逻辑回归（手写简化版）: 梯度下降训练二分类。"""


class LogisticRegressionScratch:
    """教学版逻辑回归。"""

    def __init__(self, learning_rate=0.1, epochs=2000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = 0.0

    def _sigmoid(self, z):
        import numpy as np

        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, x, y):
        import numpy as np

        n_samples, n_features = x.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for _ in range(self.epochs):
            logits = x @ self.weights + self.bias
            probs = self._sigmoid(logits)
            grad_w = (x.T @ (probs - y)) / n_samples
            grad_b = float((probs - y).mean())
            self.weights -= self.learning_rate * grad_w
            self.bias -= self.learning_rate * grad_b

    def predict_proba(self, x):
        return self._sigmoid(x @ self.weights + self.bias)

    def predict(self, x, threshold=0.5):
        import numpy as np

        return np.where(self.predict_proba(x) >= threshold, 1, 0)


def main():
    import numpy as np

    rng = np.random.default_rng(42)
    x0 = rng.normal(loc=[-1.0, -1.0], scale=[0.8, 0.8], size=(120, 2))
    x1 = rng.normal(loc=[1.2, 1.0], scale=[0.8, 0.8], size=(120, 2))
    x = np.vstack([x0, x1])
    y = np.array([0] * 120 + [1] * 120)

    model = LogisticRegressionScratch(learning_rate=0.08, epochs=2500)
    model.fit(x, y)
    pred = model.predict(x)
    acc = float((pred == y).mean())

    print("=== Logistic Regression (scratch) ===")
    print(f"训练准确率={acc:.4f}")


if __name__ == "__main__":
    main()
