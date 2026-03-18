"""PCA（sklearn 版）: 主成分分析降维。"""


def main():
    try:
        from sklearn.datasets import load_iris
        from sklearn.decomposition import PCA
    except ImportError:
        print("缺少依赖，请安装: pip install scikit-learn")
        return

    iris = load_iris()
    x = iris.data
    model = PCA(n_components=2, random_state=42)
    x_2d = model.fit_transform(x)
    print("=== PCA (sklearn) ===")
    print(f"降维后形状: {x_2d.shape}, 方差贡献率: {model.explained_variance_ratio_}")


if __name__ == "__main__":
    main()
