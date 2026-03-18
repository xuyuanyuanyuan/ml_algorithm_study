"""PCA（手写简化版）: 协方差矩阵特征分解。"""


class PCAScratch:
    """教学版 PCA。"""

    def __init__(self, n_components=2):
        self.n_components = n_components
        self.components_ = None
        self.mean_ = None

    def fit(self, x):
        import numpy as np

        self.mean_ = x.mean(axis=0)
        x_center = x - self.mean_
        cov = np.cov(x_center.T)
        eig_vals, eig_vecs = np.linalg.eigh(cov)
        idx = np.argsort(eig_vals)[::-1]
        self.components_ = eig_vecs[:, idx[: self.n_components]]

    def transform(self, x):
        x_center = x - self.mean_
        return x_center @ self.components_

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)


def main():
    import numpy as np

    rng = np.random.default_rng(42)
    x = rng.normal(size=(150, 5))
    x[:, 2] = 0.6 * x[:, 0] - 0.4 * x[:, 1] + rng.normal(0, 0.1, size=150)

    pca = PCAScratch(n_components=2)
    x_2d = pca.fit_transform(x)
    print("=== PCA (scratch) ===")
    print(f"降维后形状: {x_2d.shape}")


if __name__ == "__main__":
    main()
