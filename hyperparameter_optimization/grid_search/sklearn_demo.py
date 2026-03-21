"""网格搜索 Grid Search（sklearn 版）：使用 GridSearchCV 穷举参数组合。"""


def main():
    # 依赖说明：
    # - scikit-learn：数据集、模型、网格搜索
    try:
        from sklearn.datasets import load_breast_cancer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score
        from sklearn.model_selection import GridSearchCV, train_test_split
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
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

    # 2. 建立“标准化 + 逻辑回归”的完整流程。
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(random_state=42)),
        ]
    )

    # 3. 准备参数网格。
    # Grid Search 的核心思想是：把每个候选参数都试一遍。
    param_grid = {
        "model__C": [0.01, 0.1, 1.0, 10.0],
        "model__max_iter": [300, 800],
        "model__solver": ["lbfgs"],
    }

    # 4. 启动网格搜索。
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        scoring="accuracy",
    )
    search.fit(x_train, y_train)

    # 5. 用最优模型在测试集上评估。
    best_model = search.best_estimator_
    y_pred = best_model.predict(x_test)
    test_acc = accuracy_score(y_test, y_pred)

    print("=== GridSearchCV Demo ===")
    print(f"尝试参数组合数量：{len(search.cv_results_['params'])}")
    print(f"最优参数：{search.best_params_}")
    print(f"最佳交叉验证准确率：{search.best_score_:.4f}")
    print(f"测试集准确率：{test_acc:.4f}")


if __name__ == "__main__":
    main()
