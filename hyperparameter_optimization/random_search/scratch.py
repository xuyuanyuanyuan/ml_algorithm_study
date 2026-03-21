"""随机搜索（手写最小版）：随机采样若干组参数并比较验证集表现。"""


def sample_random_params(search_space, rng):
    """从参数空间中随机采样一组参数。"""
    return {
        "n_estimators": int(rng.choice(search_space["n_estimators"])),
        "max_depth": search_space["max_depth"][int(rng.integers(len(search_space["max_depth"])))],
        "min_samples_split": int(rng.choice(search_space["min_samples_split"])),
        "min_samples_leaf": int(rng.choice(search_space["min_samples_leaf"])),
    }


def build_model(params):
    """根据随机采样出来的参数创建模型。"""
    from sklearn.ensemble import RandomForestClassifier

    return RandomForestClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        min_samples_split=params["min_samples_split"],
        min_samples_leaf=params["min_samples_leaf"],
        random_state=42,
    )


def manual_random_search(x_train, y_train, x_val, y_val, search_space, n_trials=10, seed=42):
    """
    手写最小随机搜索流程：
    - 定义参数采样空间
    - 随机采样若干组参数
    - 训练并在验证集评估
    - 返回最优参数
    """
    from sklearn.metrics import accuracy_score

    import numpy as np

    rng = np.random.default_rng(seed)
    best_params = None
    best_score = -1.0

    for trial in range(1, n_trials + 1):
        params = sample_random_params(search_space, rng)
        model = build_model(params)
        model.fit(x_train, y_train)

        y_val_pred = model.predict(x_val)
        val_score = accuracy_score(y_val, y_val_pred)
        print(f"第 {trial:2d} 次采样：{params} -> 验证集准确率={val_score:.4f}")

        if val_score > best_score:
            best_score = val_score
            best_params = params

    return best_params, best_score


def main():
    # 依赖说明：
    # - numpy：随机采样
    # - scikit-learn：数据集、模型、指标
    try:
        from sklearn.datasets import load_breast_cancer
        from sklearn.metrics import accuracy_score
        from sklearn.model_selection import train_test_split
    except ImportError:
        print("缺少依赖，请安装：pip install numpy scikit-learn")
        return

    import numpy as np

    # 1. 准备数据。
    data = load_breast_cancer()
    x_train_val, x_test, y_train_val, y_test = train_test_split(
        data.data,
        data.target,
        test_size=0.2,
        random_state=42,
        stratify=data.target,
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val,
        y_train_val,
        test_size=0.25,
        random_state=42,
        stratify=y_train_val,
    )

    # 2. 定义随机搜索空间。
    search_space = {
        "n_estimators": [50, 80, 120, 160, 220],
        "max_depth": [3, 5, 7, None],
        "min_samples_split": [2, 4, 6, 8],
        "min_samples_leaf": [1, 2, 4],
    }

    # 3. 手写随机搜索。
    best_params, best_val_score = manual_random_search(
        x_train,
        y_train,
        x_val,
        y_val,
        search_space,
        n_trials=10,
        seed=42,
    )

    # 4. 用最优参数在训练集 + 验证集上重新训练，再去测试集评估。
    best_model = build_model(best_params)
    x_final_train = np.vstack([x_train, x_val])
    y_final_train = np.hstack([y_train, y_val])
    best_model.fit(x_final_train, y_final_train)

    y_test_pred = best_model.predict(x_test)
    test_acc = accuracy_score(y_test, y_test_pred)

    print("\n=== 手写 Random Search ===")
    print(f"最优参数：{best_params}")
    print(f"最佳验证集准确率：{best_val_score:.4f}")
    print(f"测试集准确率：{test_acc:.4f}")


if __name__ == "__main__":
    main()
