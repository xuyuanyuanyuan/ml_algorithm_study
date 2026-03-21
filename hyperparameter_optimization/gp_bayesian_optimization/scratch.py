"""高斯过程贝叶斯优化（教学型简化版）：用简化 surrogate + acquisition 展示核心流程。"""


def evaluate_logistic_regression(log10_c, x_train, y_train, x_val, y_val):
    """
    把 log10(C) 映射回真实的 C，再训练逻辑回归。

    说明：
    - C 越大，正则化越弱
    - 这里把超参数优化问题转成一维连续搜索，更适合教学展示
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    c_value = 10 ** float(log10_c)
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(C=c_value, max_iter=600, solver="lbfgs", random_state=42)),
        ]
    )
    model.fit(x_train, y_train)
    y_val_pred = model.predict(x_val)
    return accuracy_score(y_val, y_val_pred)


def surrogate_predict(candidate, history_x, history_y, bandwidth=0.45):
    """
    一个非常简化的 surrogate model。

    真正的 GP Bayesian Optimization 会用高斯过程回归来预测：
    - 某个候选点的期望表现
    - 以及这个位置的不确定性

    这里为了教学更直观，使用核加权平均来近似“预测均值”，
    再用“附近观测点是否稀少”来近似“不确定性”。
    """
    import numpy as np

    history_x = np.asarray(history_x, dtype=float)
    history_y = np.asarray(history_y, dtype=float)
    distances = history_x - float(candidate)
    weights = np.exp(-0.5 * (distances / bandwidth) ** 2)
    weight_sum = float(weights.sum())

    if weight_sum < 1e-12:
        mean = float(history_y.mean())
    else:
        mean = float(np.dot(weights, history_y) / weight_sum)

    uncertainty = float(1.0 / np.sqrt(weight_sum + 1e-8))
    return mean, uncertainty


def acquisition_ucb(mean, uncertainty, beta=0.12):
    """
    acquisition function（采集函数）。

    它负责回答一个问题：
    - 下一次应该试哪里？

    这里使用最常见的思路之一 UCB：
    - mean：更偏向“利用”目前看起来好的区域
    - uncertainty：更偏向“探索”还不确定的区域
    """
    return mean + beta * uncertainty


def simplified_gp_bayes_search(x_train, y_train, x_val, y_val, n_calls=14):
    """运行一个教学型简化版贝叶斯优化。"""
    import numpy as np

    candidate_grid = np.linspace(-3.0, 2.0, 121)
    history_x = []
    history_y = []

    # 先做几个初始点，模拟真实贝叶斯优化中的 warm start。
    initial_points = [-2.5, -1.0, 0.5]
    for point in initial_points:
        score = evaluate_logistic_regression(point, x_train, y_train, x_val, y_val)
        history_x.append(point)
        history_y.append(score)
        print(f"初始点评估：log10(C)={point:.2f} -> 验证集准确率={score:.4f}")

    while len(history_x) < n_calls:
        best_candidate = None
        best_acquisition = -float("inf")

        for candidate in candidate_grid:
            if any(abs(candidate - observed) < 1e-12 for observed in history_x):
                continue

            mean, uncertainty = surrogate_predict(candidate, history_x, history_y)
            acquisition_value = acquisition_ucb(mean, uncertainty)

            if acquisition_value > best_acquisition:
                best_acquisition = acquisition_value
                best_candidate = candidate

        score = evaluate_logistic_regression(best_candidate, x_train, y_train, x_val, y_val)
        history_x.append(float(best_candidate))
        history_y.append(float(score))

        print(
            f"新增采样：log10(C)={best_candidate:.2f} "
            f"| acquisition={best_acquisition:.4f} | 验证集准确率={score:.4f}"
        )

    best_index = int(np.argmax(history_y))
    return history_x[best_index], history_y[best_index], history_x, history_y


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

    # 1. 准备数据。
    data = load_breast_cancer()
    x_train, x_val, y_train, y_val = train_test_split(
        data.data,
        data.target,
        test_size=0.2,
        random_state=42,
        stratify=data.target,
    )

    # 2. 运行教学型简化版 GP Bayesian Optimization。
    best_log10_c, best_score, history_x, history_y = simplified_gp_bayes_search(
        x_train,
        y_train,
        x_val,
        y_val,
        n_calls=14,
    )

    print("\n=== 简化版 GP Bayesian Optimization ===")
    print("说明：这里不是工业级高斯过程优化器，而是用核加权 surrogate 模拟原理。")
    print(f"历史评估次数：{len(history_x)}")
    print(f"最优 log10(C)：{best_log10_c:.4f}")
    print(f"最优 C：{10 ** best_log10_c:.6f}")
    print(f"最佳验证集准确率：{best_score:.4f}")


if __name__ == "__main__":
    main()
