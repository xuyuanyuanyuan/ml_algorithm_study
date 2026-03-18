"""层次聚类（sklearn 版）: 自底向上合并样本簇。"""


def main():
    try:
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.datasets import make_blobs
    except ImportError:
        print("缺少依赖，请安装: pip install scikit-learn")
        return

    x, _ = make_blobs(n_samples=180, centers=3, cluster_std=0.9, random_state=42)
    model = AgglomerativeClustering(n_clusters=3, linkage="ward")
    labels = model.fit_predict(x)
    print("=== Hierarchical Clustering (sklearn) ===")
    print(f"簇标签前10个: {labels[:10]}")


if __name__ == "__main__":
    main()
