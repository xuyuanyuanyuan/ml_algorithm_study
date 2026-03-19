"""AdaBoost（sklearn 版）: Boosting 思想下的加权集成。"""


def main():
    try:
        from sklearn.datasets import load_breast_cancer
        from sklearn.ensemble import AdaBoostClassifier
        from sklearn.metrics import accuracy_score
        from sklearn.model_selection import train_test_split
        from sklearn.tree import DecisionTreeClassifier
    except ImportError:
        print("缺少依赖，请安装: pip install scikit-learn")
        return

    data = load_breast_cancer()
    x_train, x_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42, stratify=data.target
    )

    # sklearn 新旧版本参数名不同，做兼容处理。
    stump = DecisionTreeClassifier(max_depth=1, random_state=42)
    try:
        model = AdaBoostClassifier(
            estimator=stump, n_estimators=120, learning_rate=0.8, random_state=42
        )
    except TypeError:
        model = AdaBoostClassifier(
            base_estimator=stump, n_estimators=120, learning_rate=0.8, random_state=42
        )

    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    acc = accuracy_score(y_test, pred)

    print("=== AdaBoost (sklearn) ===")
    print(f"准确率={acc:.4f}")


if __name__ == "__main__":
    main()
