"""随机搜索 Random Search（sklearn 版）：使用 RandomizedSearchCV 随机采样参数。"""


def main():
    # 依赖说明：
    # - scikit-learn：数据集、模型、随机搜索
    try:
        from sklearn.datasets import load_breast_cancer
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score
        from sklearn.model_selection import RandomizedSearchCV, train_test_split
    except ImportError:
        print("缺少依赖，请安装：pip install scikit-learn")
        return

    # 1. 准备数据。
    data = load_breast_cancer()
    x_train, x_test, y_train, y_test = train_test_split(
        data.data,
        data.target,
        test_size=0.2,
        random_state=42,
        stratify=data.target,
    )

    # 2. 定义参数采样空间。
    param_distributions = {
        "n_estimators": [50, 80, 120, 160, 220],
        "max_depth": [3, 5, 7, None],
        "min_samples_split": [2, 4, 6, 8],
        "min_samples_leaf": [1, 2, 4],
    }

    # 3. 启动随机搜索。
    model = RandomForestClassifier(random_state=42)
    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_distributions,
        n_iter=10,
        cv=5,
        scoring="accuracy",
        random_state=42,
    )
    search.fit(x_train, y_train)

    # 4. 评估最优模型。
    best_model = search.best_estimator_
    y_pred = best_model.predict(x_test)
    test_acc = accuracy_score(y_test, y_pred)

    print("=== RandomizedSearchCV Demo ===")
    print(f"随机尝试次数：{search.n_iter}")
    print(f"最优参数：{search.best_params_}")
    print(f"最佳交叉验证准确率：{search.best_score_:.4f}")
    print(f"测试集准确率：{test_acc:.4f}")


if __name__ == "__main__":
    main()
