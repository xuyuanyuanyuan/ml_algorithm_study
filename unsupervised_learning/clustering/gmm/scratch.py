"""GMM（手写简化版）: EM 算法的二维演示。"""


class GMMScratch:
    """教学版 GMM，固定协方差为对角矩阵，便于理解。"""

    def __init__(self, n_components=2, epochs=40):
        self.n_components = n_components
        self.epochs = epochs
        self.means = None
        self.vars = None
        self.weights = None

    def _gaussian_pdf(self, x, mean, var):
        import numpy as np

        coeff = 1.0 / np.sqrt((2 * np.pi) ** len(x) * np.prod(var))
        exp_term = np.exp(-0.5 * np.sum(((x - mean) ** 2) / var))
        return coeff * exp_term

    def fit(self, x_data):
        import numpy as np

        n_samples, n_features = x_data.shape
        rng = np.random.default_rng(42)
        self.means = x_data[rng.choice(n_samples, self.n_components, replace=False)]
        self.vars = np.ones((self.n_components, n_features))
        self.weights = np.ones(self.n_components) / self.n_components

        for _ in range(self.epochs):
            # E 步: 计算每个样本属于每个高斯分量的责任度。
            resp = np.zeros((n_samples, self.n_components))
            for i in range(n_samples):
                for k in range(self.n_components):
                    resp[i, k] = self.weights[k] * self._gaussian_pdf(
                        x_data[i], self.means[k], self.vars[k]
                    )
                resp[i] /= resp[i].sum() + 1e-12

            # M 步: 更新参数。
            nk = resp.sum(axis=0)
            self.weights = nk / n_samples
            self.means = (resp.T @ x_data) / nk[:, None]
            for k in range(self.n_components):
                diff = x_data - self.means[k]
                self.vars[k] = (resp[:, [k]] * (diff**2)).sum(axis=0) / nk[k] + 1e-6

    def predict(self, x_data):
        import numpy as np

        labels = []
        for x in x_data:
            probs = [
                self.weights[k] * self._gaussian_pdf(x, self.means[k], self.vars[k])
                for k in range(self.n_components)
            ]
            labels.append(int(np.argmax(probs)))
        return np.array(labels)


def main():
    import numpy as np

    rng = np.random.default_rng(42)
    x1 = rng.normal(loc=[0, 0], scale=[0.6, 0.6], size=(90, 2))
    x2 = rng.normal(loc=[3, 3], scale=[0.7, 0.7], size=(90, 2))
    x = np.vstack([x1, x2])

    model = GMMScratch(n_components=2, epochs=35)
    model.fit(x)
    labels = model.predict(x)
    print("=== GMM (scratch) ===")
    print(f"标签前10个: {labels[:10]}")


if __name__ == "__main__":
    main()
