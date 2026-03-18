"""t-SNE（sklearn 版）: 高维数据可视化到 2D。"""


def main():
    # 关键参数: perplexity 表示局部邻域规模。
    try:
        from sklearn.datasets import load_digits
        from sklearn.manifold import TSNE
    except ImportError:
        print("缺少依赖，请安装: pip install scikit-learn")
        return

    data = load_digits()
    x = data.data[:500]
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, init="pca")
    x_2d = tsne.fit_transform(x)
    print("=== t-SNE (sklearn) ===")
    print(f"降维后形状: {x_2d.shape}")


if __name__ == "__main__":
    main()
