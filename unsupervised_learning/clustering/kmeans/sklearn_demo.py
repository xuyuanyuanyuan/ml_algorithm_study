"""K-Means 聚类（scikit-learn 版）：自动造二维数据并可视化结果。"""


def _get_styled_pyplot():
    """获取统一风格的 pyplot。"""
    try:
        import sys
        from pathlib import Path

        project_root = None
        for parent in Path(__file__).resolve().parents:
            if (parent / "utils" / "plot_utils.py").exists():
                project_root = parent
                break
        if project_root is None:
            return None
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        from utils.plot_utils import get_styled_pyplot
    except Exception:
        return None
    return get_styled_pyplot()


def main():
    # 1. 导入依赖库。
    try:
        from sklearn.cluster import KMeans
        from sklearn.datasets import make_blobs
    except ImportError:
        print("缺少依赖。请安装：pip install scikit-learn")
        return

    # 2. 自动生成二维聚类数据。
    X, _ = make_blobs(
        n_samples=240,
        centers=3,
        n_features=2,
        cluster_std=0.9,
        random_state=42,
    )

    # 3. 创建并训练 KMeans 模型。
    model = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = model.fit_predict(X)
    centers = model.cluster_centers_

    # 4. 输出结果信息。
    print("=== KMeans（sklearn）===")
    print(f"簇中心：\n{centers}")
    print(f"惯性（inertia）：{model.inertia_:.4f}")

    # 5. 可视化聚类结果。
    plt = _get_styled_pyplot()
    if plt is None:
        print("未安装 matplotlib，跳过绘图。可安装：pip install matplotlib")
        return

    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="tab10", s=38, alpha=0.75)
    plt.scatter(
        centers[:, 0],
        centers[:, 1],
        c="tab:red",
        marker="X",
        s=220,
        label="簇中心",
    )
    plt.title("KMeans 聚类结果")
    plt.xlabel("特征 1")
    plt.ylabel("特征 2")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
