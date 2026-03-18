"""XGBoost（库版本）: 梯度提升树的高性能实现。"""


def main():
    # 关键参数: n_estimators 树数量, max_depth 树深, learning_rate 学习率。
    try:
        from sklearn.datasets import load_breast_cancer
        from sklearn.metrics import accuracy_score
        from sklearn.model_selection import train_test_split
        from xgboost import XGBClassifier
    except ImportError:
        print("缺少依赖，请安装: pip install xgboost scikit-learn")
        return

    data = load_breast_cancer()
    x_train, x_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42, stratify=data.target
    )
    model = XGBClassifier(
        n_estimators=120,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    acc = accuracy_score(y_test, pred)
    print("=== XGBoost Demo ===")
    print(f"准确率={acc:.4f}")


if __name__ == "__main__":
    main()
