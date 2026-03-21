"""TPE（教学型简化版）：展示“好样本 / 坏样本分布”如何指导下一次采样。"""


def evaluate_logistic_regression(log10_c, x_train, y_train, x_val, y_val):
    """评估一组超参数对应的验证集表现。"""
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


def gaussian_kde_density(x_value, samples, bandwidth=0.35):
    """
    用最简核密度估计近似分布。

    真正的 TPE 会分别拟合：
    - 好样本分布 l(x)
    - 坏样本分布 g(x)

    然后倾向选择 l(x) / g(x) 比值较大的点。
    """
    import numpy as np

    if len(samples) == 0:
        return 1e-8

    samples = np.asarray(samples, dtype=float)
    diff = (x_value - samples) / bandwidth
    values = np.exp(-0.5 * diff**2)
    return float(values.mean() + 1e-8)


def split_good_bad(history, gamma=0.3):
    """按分数把历史试验拆成“好样本”和“坏样本”。"""
    history = sorted(history, key=lambda item: item["score"], reverse=True)
    n_good = max(2, int(len(history) * gamma))
    good = history[:n_good]
    bad = history[n_good:]
    return good, bad


def sample_from_good_region(good_samples, rng, low=-3.0, high=2.0):
    """优先在好样本附近采样，体现 TPE 的搜索偏好。"""
    if not good_samples:
        return float(rng.uniform(low, high))

    center = float(rng.choice(good_samples))
    candidate = center + float(rng.normal(0.0, 0.35))
    return float(min(max(candidate, low), high))


def simplified_tpe_search(x_train, y_train, x_val, y_val, n_trials=16, seed=42):
    """运行教学型简化版 TPE。"""
    import numpy as np

    rng = np.random.default_rng(seed)
    history = []

    # 先随机做一些 warm start。
    for _ in range(5):
        log10_c = float(rng.uniform(-3.0, 2.0))
        score = evaluate_logistic_regression(log10_c, x_train, y_train, x_val, y_val)
        history.append({"log10_c": log10_c, "score": score})
        print(f"初始试验：log10(C)={log10_c:.4f} -> 验证集准确率={score:.4f}")

    while len(history) < n_trials:
        good, bad = split_good_bad(history, gamma=0.3)
        good_samples = [item["log10_c"] for item in good]
        bad_samples = [item["log10_c"] for item in bad]

        best_candidate = None
        best_ratio = -float("inf")

        # 从好样本附近采一批候选点，再比较 l(x)/g(x)。
        for _ in range(60):
            candidate = sample_from_good_region(good_samples, rng)
            l_value = gaussian_kde_density(candidate, good_samples)
            g_value = gaussian_kde_density(candidate, bad_samples)
            ratio = l_value / g_value

            if ratio > best_ratio:
                best_ratio = ratio
                best_candidate = candidate

        score = evaluate_logistic_regression(best_candidate, x_train, y_train, x_val, y_val)
        history.append({"log10_c": float(best_candidate), "score": float(score)})
        print(
            f"新增试验：log10(C)={best_candidate:.4f} | l/g={best_ratio:.4f} "
            f"-> 验证集准确率={score:.4f}"
        )

    best_result = max(history, key=lambda item: item["score"])
    return best_result, history


def main():
    # 依赖说明：
    # - numpy：采样与数值计算
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

    # 2. 运行教学型简化版 TPE。
    best_result, history = simplified_tpe_search(
        x_train,
        y_train,
        x_val,
        y_val,
        n_trials=16,
        seed=42,
    )

    print("\n=== 简化版 TPE ===")
    print("说明：这里不是完整工业级 TPE，而是用核密度估计模拟好/坏样本分布思想。")
    print(f"历史试验次数：{len(history)}")
    print(f"最优 log10(C)：{best_result['log10_c']:.4f}")
    print(f"最优 C：{10 ** best_result['log10_c']:.6f}")
    print(f"最佳验证集准确率：{best_result['score']:.4f}")


if __name__ == "__main__":
    main()
