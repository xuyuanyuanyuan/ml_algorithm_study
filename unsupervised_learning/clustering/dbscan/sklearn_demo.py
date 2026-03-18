"""DBSCAN（sklearn 版）: 基于密度的聚类。"""


def main():
    # 关键参数: eps 邻域半径, min_samples 核心点最少样本数。
    try:
        from sklearn.cluster import DBSCAN
        from sklearn.datasets import make_moons
    except ImportError:
        print("缺少依赖，请安装: pip install scikit-learn")
        return

    x, _ = make_moons(n_samples=250, noise=0.08, random_state=42)
    model = DBSCAN(eps=0.2, min_samples=5)
    labels = model.fit_predict(x)
    noise_count = int((labels == -1).sum())
    print("=== DBSCAN (sklearn) ===")
    print(f"噪声点数量={noise_count}, 标签前10个={labels[:10]}")


if __name__ == "__main__":
    main()
