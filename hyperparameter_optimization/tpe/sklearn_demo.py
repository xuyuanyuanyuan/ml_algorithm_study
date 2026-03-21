"""TPE（库版）：使用 Optuna 的 TPESampler 搜索超参数。"""


def main():
    # 依赖说明：
    # - scikit-learn：数据集、模型、指标
    # - optuna：TPE 采样器
    try:
        import optuna
        from optuna.samplers import TPESampler
        from sklearn.datasets import load_iris
        from sklearn.metrics import accuracy_score
        from sklearn.model_selection import cross_val_score, train_test_split
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC
    except ImportError:
        print("缺少依赖，请安装：pip install scikit-learn optuna")
        return

    data = load_iris()
    x_train, x_test, y_train, y_test = train_test_split(
        data.data,
        data.target,
        test_size=0.2,
        random_state=42,
        stratify=data.target,
    )

    def objective(trial):
        c_value = trial.suggest_float("C", 1e-2, 100.0, log=True)
        gamma_value = trial.suggest_float("gamma", 1e-3, 1.0, log=True)

        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("svc", SVC(kernel="rbf", C=c_value, gamma=gamma_value)),
            ]
        )
        score = cross_val_score(model, x_train, y_train, cv=5, scoring="accuracy").mean()
        return float(score)

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=20, show_progress_bar=False)

    best_params = study.best_params
    best_model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svc", SVC(kernel="rbf", C=best_params["C"], gamma=best_params["gamma"])),
        ]
    )
    best_model.fit(x_train, y_train)
    y_test_pred = best_model.predict(x_test)
    test_acc = accuracy_score(y_test, y_test_pred)

    print("=== TPE / Optuna Demo ===")
    print(f"试验次数：{len(study.trials)}")
    print(f"最优参数：{best_params}")
    print(f"最佳交叉验证准确率：{study.best_value:.4f}")
    print(f"测试集准确率：{test_acc:.4f}")


if __name__ == "__main__":
    main()
