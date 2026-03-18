"""CatBoost（库版本）: 对类别特征处理友好的提升模型。"""


def main():
    try:
        from catboost import CatBoostClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
    except ImportError:
        print("缺少依赖，请安装: pip install catboost scikit-learn")
        return

    # 构造一个包含类别特征的小数据集。
    x = [
        ["A", "高", 25], ["A", "中", 30], ["B", "低", 22], ["B", "高", 40],
        ["C", "中", 35], ["C", "低", 28], ["A", "低", 45], ["B", "中", 33],
        ["C", "高", 29], ["A", "中", 38], ["B", "低", 27], ["C", "中", 31],
    ]
    y = [0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=42
    )

    model = CatBoostClassifier(
        iterations=80, depth=4, learning_rate=0.1, verbose=False, random_state=42
    )
    model.fit(x_train, y_train, cat_features=[0, 1])
    pred = model.predict(x_test)
    acc = accuracy_score(y_test, pred)
    print("=== CatBoost Demo ===")
    print(f"准确率={acc:.4f}")


if __name__ == "__main__":
    main()
