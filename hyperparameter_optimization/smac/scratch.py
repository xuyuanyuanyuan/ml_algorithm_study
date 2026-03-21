"""SMAC（教学型简化版）：用 surrogate + acquisition + sequential search 演示 SMBO 思想。"""


def sample_random_config(rng):
    """从搜索空间中随机采样一组配置。"""
    return {
        "log10_C": float(rng.uniform(-3.0, 2.0)),
        "max_iter": int(rng.integers(300, 1201)),
    }


def config_to_features(config):
    """
    把超参数配置映射成 surrogate 模型使用的特征。

    工业版 SMAC 常用随机森林做 surrogate；
    这里为了教学更直观，使用一个简单的二次多项式近似。
    """
    x1 = float(config["log10_C"])
    x2 = float(config["max_iter"])
    x2_scaled = x2 / 1000.0
    return [1.0, x1, x2_scaled, x1 * x1, x2_scaled * x2_scaled, x1 * x2_scaled]


def fit_surrogate(history):
    """拟合一个最小 surrogate model。"""
    import numpy as np

    x_matrix = np.asarray([config_to_features(item["config"]) for item in history], dtype=float)
    y_vector = np.asarray([item["score"] for item in history], dtype=float)
    coef, _, _, _ = np.linalg.lstsq(x_matrix, y_vector, rcond=None)
    return coef


def predict_surrogate(config, coef):
    """使用 surrogate model 预测某个候选配置的表现。"""
    import numpy as np

    features = np.asarray(config_to_features(config), dtype=float)
    return float(features @ coef)


def exploration_bonus(config, history):
    """
    给离历史样本较远的点更高探索奖励。

    这相当于 acquisition function 中“鼓励探索”的那一部分。
    """
    import numpy as np

    current = np.asarray([config["log10_C"], config["max_iter"] / 1000.0], dtype=float)
    observed = np.asarray(
        [[item["config"]["log10_C"], item["config"]["max_iter"] / 1000.0] for item in history],
        dtype=float,
    )
    distances = np.sqrt(((observed - current) ** 2).sum(axis=1))
    return float(distances.min())


def acquisition(config, coef, history, kappa=0.08):
    """
    acquisition function = surrogate 预测值 + 探索奖励。

    说明：
    - surrogate：估计这个点可能有多好
    - exploration_bonus：估计这个点是否值得再探索
    """
    predicted_score = predict_surrogate(config, coef)
    explore = exploration_bonus(config, history)
    return predicted_score + kappa * explore


def evaluate_config(config, x_train, y_train, x_val, y_val):
    """评估一个具体超参数配置。"""
    from sklearn.metrics import accuracy_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    c_value = 10 ** float(config["log10_C"])
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "svc",
                SVC(
                    kernel="linear",
                    C=c_value,
                    max_iter=int(config["max_iter"]),
                    random_state=42,
                ),
            ),
        ]
    )
    model.fit(x_train, y_train)
    y_val_pred = model.predict(x_val)
    return accuracy_score(y_val, y_val_pred)


def smac_like_search(x_train, y_train, x_val, y_val, n_trials=16, seed=42):
    """
    一个教学型简化版 SMAC 搜索流程。

    主线思路：
    1. 先随机试几个点
    2. 用历史数据训练 surrogate model
    3. 用 acquisition function 选下一组超参数
    4. 评估后再更新历史
    5. 重复以上步骤
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    history = []

    # 初始随机试验。
    for _ in range(5):
        config = sample_random_config(rng)
        score = evaluate_config(config, x_train, y_train, x_val, y_val)
        history.append({"config": config, "score": score})
        print(f"初始试验：{config} -> 验证集准确率={score:.4f}")

    while len(history) < n_trials:
        coef = fit_surrogate(history)

        best_candidate = None
        best_acquisition = -float("inf")

        # 生成一批候选点，再让 acquisition function 去挑最好的一组。
        for _ in range(120):
            candidate = sample_random_config(rng)
            value = acquisition(candidate, coef, history)
            if value > best_acquisition:
                best_candidate = candidate
                best_acquisition = value

        score = evaluate_config(best_candidate, x_train, y_train, x_val, y_val)
        history.append({"config": best_candidate, "score": score})
        print(
            f"新增试验：{best_candidate} | acquisition={best_acquisition:.4f} "
            f"-> 验证集准确率={score:.4f}"
        )

    best_result = max(history, key=lambda item: item["score"])
    return best_result, history


def main():
    # 依赖说明：
    # - numpy：数值计算
    # - scikit-learn：数据集、模型、指标
    try:
        from sklearn.datasets import load_breast_cancer
        from sklearn.model_selection import train_test_split
    except ImportError:
        print("缺少依赖，请安装：pip install numpy scikit-learn")
        return

    data = load_breast_cancer()
    x_train, x_val, y_train, y_val = train_test_split(
        data.data,
        data.target,
        test_size=0.2,
        random_state=42,
        stratify=data.target,
    )

    best_result, history = smac_like_search(
        x_train,
        y_train,
        x_val,
        y_val,
        n_trials=16,
        seed=42,
    )

    print("\n=== 简化版 SMAC ===")
    print("说明：工业版 SMAC 更常使用随机森林 surrogate，这里用二次多项式做教学近似。")
    print(f"历史试验次数：{len(history)}")
    print(f"最优参数：{best_result['config']}")
    print(f"最佳验证集准确率：{best_result['score']:.4f}")


if __name__ == "__main__":
    main()
