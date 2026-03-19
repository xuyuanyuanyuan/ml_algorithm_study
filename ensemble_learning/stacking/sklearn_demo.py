"""Stacking（sklearn 版）: 多个基学习器 + 元学习器。"""


def main():
    try:
        from sklearn.datasets import load_breast_cancer
        from sklearn.ensemble import RandomForestClassifier, StackingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score
        from sklearn.model_selection import train_test_split
        from sklearn.neighbors import KNeighborsClassifier
    except ImportError:
        print("缺少依赖，请安装: pip install scikit-learn")
        return

    data = load_breast_cancer()
    x_train, x_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42, stratify=data.target
    )

    estimators = [
        ("rf", RandomForestClassifier(n_estimators=80, random_state=42)),
        ("knn", KNeighborsClassifier(n_neighbors=7)),
    ]
    model = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=1000),
        stack_method="predict_proba",
    )
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    acc = accuracy_score(y_test, pred)

    print("=== Stacking (sklearn) ===")
    print(f"准确率={acc:.4f}")


if __name__ == "__main__":
    main()
