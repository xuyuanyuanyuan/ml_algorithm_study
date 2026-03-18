"""决策树分类（scikit-learn 版）：使用 iris 并输出准确率。"""


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
        from sklearn.tree import DecisionTreeClassifier, plot_tree
    except ImportError:
        print("缺少依赖。请安装：pip install scikit-learn")
        return

    # 2. 加载 iris 数据集。
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

    # 4. 创建并训练决策树模型。
    model = DecisionTreeClassifier(max_depth=3, random_state=42)
    model.fit(X_train, y_train)

    # 5. 预测并计算准确率。
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("=== 决策树（sklearn）===")
    print(f"测试集准确率：{acc:.4f}")

    # 6. 可视化决策树结构。
    plt = _get_styled_pyplot()
    if plt is None:
        print("未安装 matplotlib，跳过绘图。可安装：pip install matplotlib")
        return

    plt.figure(figsize=(10, 6))
    plot_tree(
        model,
        feature_names=iris.feature_names,
        class_names=iris.target_names,
        filled=True,
        rounded=True,
    )
    plt.title("DecisionTreeClassifier 决策树结构")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
