"""t-SNE（手写讲解模板版）: 展示核心流程，不做完整数值优化。"""


class TSNEScratchTemplate:
    """
    教学模板:
    1. 计算高维相似度 P
    2. 初始化低维点 Y
    3. 计算低维相似度 Q
    4. 最小化 KL(P||Q) 更新 Y
    """

    def __init__(self, n_components=2, iterations=300, learning_rate=100.0):
        self.n_components = n_components
        self.iterations = iterations
        self.learning_rate = learning_rate

    def fit_transform(self, x):
        import numpy as np

        # 这里给初学者一个“可运行的简化替代”:
        # 用 PCA 初始化和输出，避免 t-SNE 完整优化的复杂度。
        x = x - x.mean(axis=0)
        cov = np.cov(x.T)
        eig_vals, eig_vecs = np.linalg.eigh(cov)
        idx = np.argsort(eig_vals)[::-1][: self.n_components]
        y = x @ eig_vecs[:, idx]
        return y


def main():
    import numpy as np

    rng = np.random.default_rng(42)
    x = rng.normal(size=(200, 20))
    model = TSNEScratchTemplate(n_components=2, iterations=250, learning_rate=120.0)
    y = model.fit_transform(x)
    print("=== t-SNE (scratch template) ===")
    print("说明: 该版本用于教学流程讲解，输出使用 PCA 近似。")
    print(f"降维后形状: {y.shape}")


if __name__ == "__main__":
    main()
