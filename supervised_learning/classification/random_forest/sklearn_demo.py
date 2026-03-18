"""随机森林（sklearn 版）: 多棵决策树投票。"""


def main():
    # 关键参数: n_estimators 为树数量, max_depth 控制树复杂度。
    try:
        from sklearn.datasets import load_iris
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score
        from sklearn.model_selection import train_test_split
    except ImportError:
        print("缺少依赖，请安装: pip install scikit-learn")
        return

    iris = load_iris()
    x_train, x_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42, stratify=iris.target
    )

    model = RandomForestClassifier(
        n_estimators=100, max_depth=4, random_state=42
    )
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    acc = accuracy_score(y_test, pred)

    print("=== Random Forest (sklearn) ===")
    print(f"准确率={acc:.4f}")


if __name__ == "__main__":
    main()
