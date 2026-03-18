"""KNN 分类（scikit-learn 版）：使用 iris 数据集并输出准确率。"""


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
        from sklearn.datasets import load_iris
        from sklearn.metrics import accuracy_score
        from sklearn.model_selection import train_test_split
        from sklearn.neighbors import KNeighborsClassifier
    except ImportError:
        print("缺少依赖。请安装：pip install scikit-learn")
        return

    # 2. 读取 iris 数据集。
    iris = load_iris()
    X, y = iris.data, iris.target

    # 3. 划分训练集和测试集。
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # 4. 创建并训练 KNN 分类器。
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)

    # 5. 在测试集上预测并计算准确率。
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("=== KNN（sklearn）===")
    print(f"测试集准确率：{acc:.4f}")

    # 6. 简单可视化：只画前两个特征（花萼长度、花萼宽度）的测试集预测结果。
    plt = _get_styled_pyplot()
    if plt is None:
        print("未安装 matplotlib，跳过绘图。可安装：pip install matplotlib")
        return

    plt.figure()
    scatter = plt.scatter(
        X_test[:, 0],
        X_test[:, 1],
        c=y_pred,
        cmap="tab10",
        s=60,
        edgecolors="black",
        alpha=0.8,
    )
    plt.title("KNN 在 iris 测试集上的预测（前两个特征）")
    plt.xlabel(iris.feature_names[0])
    plt.ylabel(iris.feature_names[1])
    plt.grid(alpha=0.3)
    plt.legend(*scatter.legend_elements(), title="预测类别")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
