"""CatBoost（手写原理模拟版）: 目标编码 + 逻辑回归。"""


class CatBoostLikeScratch:
    """
    教学版 CatBoost 思路:
    1. 对类别特征做目标均值编码
    2. 在编码后特征上训练逻辑回归
    """

    def __init__(self, learning_rate=0.1, epochs=800):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.category_stats = {}
        self.weights = None
        self.bias = 0.0

    def _fit_target_encoding(self, x_cat, y):
        from collections import defaultdict

        sums = defaultdict(float)
        cnts = defaultdict(int)
        for cat, label in zip(x_cat, y):
            sums[cat] += label
            cnts[cat] += 1
        self.category_stats = {k: sums[k] / cnts[k] for k in sums}
        self.global_mean = float(sum(y) / len(y))

    def _transform(self, x_cat, x_num):
        import numpy as np

        encoded = [self.category_stats.get(cat, self.global_mean) for cat in x_cat]
        return np.column_stack([encoded, x_num])

    def _sigmoid(self, z):
        import numpy as np

        return 1 / (1 + np.exp(-z))

    def fit(self, x_cat, x_num, y):
        import numpy as np

        self._fit_target_encoding(x_cat, y)
        x = self._transform(x_cat, x_num)
        y = np.asarray(y, dtype=float)
        self.weights = np.zeros(x.shape[1])
        self.bias = 0.0

        for _ in range(self.epochs):
            prob = self._sigmoid(x @ self.weights + self.bias)
            grad_w = (x.T @ (prob - y)) / len(y)
            grad_b = float((prob - y).mean())
            self.weights -= self.learning_rate * grad_w
            self.bias -= self.learning_rate * grad_b

    def predict(self, x_cat, x_num):
        import numpy as np

        x = self._transform(x_cat, x_num)
        prob = self._sigmoid(x @ self.weights + self.bias)
        return np.where(prob >= 0.5, 1, 0)


def main():
    import numpy as np

    x_cat = np.array(["A", "A", "B", "B", "C", "C", "A", "B", "C", "A", "B", "C"])
    x_num = np.array([25, 30, 22, 40, 35, 28, 45, 33, 29, 38, 27, 31], dtype=float)
    y = np.array([0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0], dtype=float)

    model = CatBoostLikeScratch(learning_rate=0.2, epochs=1200)
    model.fit(x_cat, x_num, y)
    pred = model.predict(x_cat, x_num)
    acc = float((pred == y).mean())
    print("=== CatBoost-like Scratch ===")
    print(f"训练准确率={acc:.4f}")


if __name__ == "__main__":
    main()
