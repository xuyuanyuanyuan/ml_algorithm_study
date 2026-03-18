"""GMM（sklearn 版）: 高斯混合模型聚类。"""


def main():
    try:
        from sklearn.datasets import make_blobs
        from sklearn.mixture import GaussianMixture
    except ImportError:
        print("缺少依赖，请安装: pip install scikit-learn")
        return

    x, _ = make_blobs(n_samples=200, centers=3, cluster_std=0.8, random_state=42)
    model = GaussianMixture(n_components=3, random_state=42)
    model.fit(x)
    labels = model.predict(x)
    print("=== GMM (sklearn) ===")
    print(f"簇标签前10个: {labels[:10]}")


if __name__ == "__main__":
    main()
