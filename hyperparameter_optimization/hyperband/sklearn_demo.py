"""Hyperband（sklearn 版）：使用 HalvingRandomSearchCV 演示资源逐步分配。"""


def main():
    # 依赖说明：
    # - scikit-learn：HalvingRandomSearchCV 需要 sklearn 实验特性模块
    try:
        from sklearn.experimental import enable_halving_search_cv  # noqa: F401
        from sklearn.datasets import load_breast_cancer
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score
        from sklearn.model_selection import HalvingRandomSearchCV, train_test_split
    except ImportError:
        print("缺少依赖，请安装：pip install scikit-learn")
        print("说明：HalvingRandomSearchCV 是 sklearn 对 successive halving / Hyperband 思想的教学入口。")
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

    # 2. 定义搜索空间。
    # 这里把 n_estimators 当作“资源”。
    param_distributions = {
        "max_depth": [3, 5, 7, None],
        "min_samples_split": [2, 4, 6, 8],
        "min_samples_leaf": [1, 2, 4],
    }

    base_model = RandomForestClassifier(random_state=42)

    # 3. 启动 HalvingRandomSearchCV。
    # 它会先给很多候选参数较少资源，再逐轮淘汰表现差的候选者。
    search = HalvingRandomSearchCV(
        estimator=base_model,
        param_distributions=param_distributions,
        factor=3,
        resource="n_estimators",
        min_resources=9,
        max_resources=81,
        random_state=42,
        scoring="accuracy",
    )
    search.fit(x_train, y_train)

    # 4. 输出结果。
    best_model = search.best_estimator_
    y_pred = best_model.predict(x_test)
    test_acc = accuracy_score(y_test, y_pred)

    print("=== Hyperband / HalvingRandomSearchCV Demo ===")
    print(f"每一轮候选数量：{search.n_candidates_}")
    print(f"每一轮使用资源：{search.n_resources_}")
    print(f"最优参数：{search.best_params_}")
    print(f"最佳交叉验证准确率：{search.best_score_:.4f}")
    print(f"测试集准确率：{test_acc:.4f}")


if __name__ == "__main__":
    main()
