"""朴素贝叶斯（sklearn 版）: 假设特征条件独立。"""


def main():
    try:
        from sklearn.datasets import load_iris
        from sklearn.metrics import accuracy_score
        from sklearn.model_selection import train_test_split
        from sklearn.naive_bayes import GaussianNB
    except ImportError:
        print("缺少依赖，请安装: pip install scikit-learn")
        return

    iris = load_iris()
    x_train, x_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42, stratify=iris.target
    )

    model = GaussianNB()
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    acc = accuracy_score(y_test, pred)

    print("=== Naive Bayes (sklearn) ===")
    print(f"准确率={acc:.4f}")


if __name__ == "__main__":
    main()
