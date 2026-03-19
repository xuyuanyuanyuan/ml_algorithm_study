"""随机森林（sklearn 版）: Bagging 思想下的多树投票。"""


def main():
    try:
        from sklearn.datasets import load_iris
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score
        from sklearn.model_selection import train_test_split
    except ImportError:
        print("缺少依赖，请安装: pip install scikit-learn")
        return

    data = load_iris()
    x_train, x_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42, stratify=data.target
    )

    model = RandomForestClassifier(n_estimators=120, max_depth=4, random_state=42)
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    acc = accuracy_score(y_test, pred)

    print("=== Random Forest (sklearn) ===")
    print(f"准确率={acc:.4f}")


if __name__ == "__main__":
    main()
