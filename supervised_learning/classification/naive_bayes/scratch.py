"""朴素贝叶斯（手写简化版）: 高斯朴素贝叶斯。"""


class GaussianNBScratch:
    """教学版高斯朴素贝叶斯。"""

    def __init__(self):
        self.classes_ = None
        self.prior_ = {}
        self.mean_ = {}
        self.var_ = {}

    def fit(self, x, y):
        import numpy as np

        self.classes_ = np.unique(y)
        for cls in self.classes_:
            x_c = x[y == cls]
            self.prior_[int(cls)] = len(x_c) / len(x)
            self.mean_[int(cls)] = x_c.mean(axis=0)
            self.var_[int(cls)] = x_c.var(axis=0) + 1e-9

    def _log_gaussian_pdf(self, x, mean, var):
        import numpy as np

        return -0.5 * np.sum(np.log(2 * np.pi * var) + ((x - mean) ** 2) / var)

    def predict(self, x):
        import numpy as np

        preds = []
        for sample in x:
            scores = {}
            for cls in self.classes_:
                cls_int = int(cls)
                scores[cls_int] = np.log(self.prior_[cls_int]) + self._log_gaussian_pdf(
                    sample, self.mean_[cls_int], self.var_[cls_int]
                )
            preds.append(max(scores.items(), key=lambda item: item[1])[0])
        return np.array(preds)


def main():
    import numpy as np

    rng = np.random.default_rng(42)
    x0 = rng.normal(loc=[0.0, 0.0], scale=[1.0, 1.2], size=(90, 2))
    x1 = rng.normal(loc=[2.0, 2.2], scale=[1.0, 1.2], size=(90, 2))
    x = np.vstack([x0, x1])
    y = np.array([0] * 90 + [1] * 90)

    model = GaussianNBScratch()
    model.fit(x, y)
    pred = model.predict(x)
    acc = float((pred == y).mean())
    print("=== Naive Bayes (scratch) ===")
    print(f"训练准确率={acc:.4f}")


if __name__ == "__main__":
    main()
