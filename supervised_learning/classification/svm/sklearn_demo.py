"""支持向量机 SVM（scikit-learn 版）：使用 iris 并输出准确率。"""


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
        import numpy as np
        from sklearn.datasets import load_iris
        from sklearn.metrics import accuracy_score
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC
    except ImportError:
        print("缺少依赖。请安装：pip install scikit-learn")
        return

    # 2. 加载 iris 数据集。
    iris = load_iris()

    # 为了便于画决策边界，这里只取前两个特征。
    X = iris.data[:, :2]
    y = iris.target

    # 3. 划分训练集和测试集。
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # 4. 构建并训练 SVM 模型。
    model = make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", C=1.0, gamma="scale"),
    )
    model.fit(X_train, y_train)

    # 5. 预测并输出准确率。
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("=== SVM（sklearn）===")
    print(f"测试集准确率：{acc:.4f}")

    # 6. 绘制决策边界。
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300),
    )
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    zz = model.predict(grid_points).reshape(xx.shape)

    plt = _get_styled_pyplot()
    if plt is None:
        print("未安装 matplotlib，跳过绘图。可安装：pip install matplotlib")
        return

    plt.figure()
    plt.contourf(xx, yy, zz, alpha=0.25, cmap="tab10")
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="tab10", edgecolors="black", s=42)
    plt.title("SVM 在 iris 数据集上的分类边界（前两个特征）")
    plt.xlabel(iris.feature_names[0])
    plt.ylabel(iris.feature_names[1])
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
