"""逻辑回归（sklearn 版）: 经典二分类基线模型。"""


def main():
    # 关键参数: C 为正则强度倒数，越小正则越强。
    c_value = 1.0

    try:
        from sklearn.datasets import make_classification
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score
        from sklearn.model_selection import train_test_split
    except ImportError:
        print("缺少依赖，请安装: pip install scikit-learn")
        return

    x, y = make_classification(
        n_samples=300, n_features=6, n_informative=4, n_redundant=0, random_state=42
    )
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression(C=c_value, max_iter=1000)
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    acc = accuracy_score(y_test, pred)

    print("=== Logistic Regression (sklearn) ===")
    print(f"C={c_value}, 准确率={acc:.4f}")


if __name__ == "__main__":
    main()
