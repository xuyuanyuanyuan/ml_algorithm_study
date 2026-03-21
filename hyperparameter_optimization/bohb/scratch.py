"""BOHB（教学型简化版）：展示 Bayesian Optimization + Hyperband 的结合思想。"""


def prepare_dataset():
    """准备训练集、验证集、测试集，并完成标准化。"""
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

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

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)
    x_test = scaler.transform(x_test)
    return x_train, x_val, x_test, y_train, y_val, y_test


def sample_random_config(rng):
    """随机采样一组配置。"""
    return {
        "log10_alpha": float(rng.uniform(-5.0, -2.0)),
        "log10_eta0": float(rng.uniform(-3.0, -0.5)),
    }


def evaluate_config(config, resource, x_train, y_train, x_val, y_val):
    """
    使用指定 budget 评估一组配置。

    这里把 resource 当作训练预算：
    - resource 小：快速筛掉差配置
    - resource 大：给有潜力的配置更多训练机会
    """
    from sklearn.linear_model import SGDClassifier
    from sklearn.metrics import accuracy_score

    model = SGDClassifier(
        loss="log_loss",
        alpha=10 ** float(config["log10_alpha"]),
        learning_rate="constant",
        eta0=10 ** float(config["log10_eta0"]),
        max_iter=max(1, int(resource)),
        tol=None,
        random_state=42,
    )
    model.fit(x_train, y_train)
    y_val_pred = model.predict(x_val)
    return accuracy_score(y_val, y_val_pred)


def kde_density(point, samples, bandwidth=(0.35, 0.30)):
    """对二维参数做一个最小核密度估计。"""
    import numpy as np

    if len(samples) == 0:
        return 1e-8

    samples = np.asarray(samples, dtype=float)
    point = np.asarray(point, dtype=float)
    diff = (samples - point) / np.asarray(bandwidth, dtype=float)
    squared_dist = np.sum(diff**2, axis=1)
    values = np.exp(-0.5 * squared_dist)
    return float(values.mean() + 1e-8)


def suggest_config_from_history(history, rng, gamma=0.3):
    """
    用历史试验做一个 TPE 风格的近似采样。

    这体现了 BOHB 中“Bayesian Optimization”的部分：
    - 历史好的配置附近更值得继续采样
    - 不是完全随机找参数
    """
    import numpy as np

    if len(history) < 8:
        return sample_random_config(rng)

    sorted_history = sorted(history, key=lambda item: item["score"], reverse=True)
    n_good = max(2, int(len(sorted_history) * gamma))
    good = sorted_history[:n_good]
    bad = sorted_history[n_good:]

    good_points = np.asarray(
        [[item["config"]["log10_alpha"], item["config"]["log10_eta0"]] for item in good],
        dtype=float,
    )
    bad_points = np.asarray(
        [[item["config"]["log10_alpha"], item["config"]["log10_eta0"]] for item in bad],
        dtype=float,
    )

    best_candidate = None
    best_ratio = -float("inf")

    for _ in range(60):
        center = good_points[int(rng.integers(len(good_points)))]
        candidate_point = center + rng.normal(loc=0.0, scale=[0.35, 0.30], size=2)
        candidate_point[0] = float(np.clip(candidate_point[0], -5.0, -2.0))
        candidate_point[1] = float(np.clip(candidate_point[1], -3.0, -0.5))

        l_value = kde_density(candidate_point, good_points)
        g_value = kde_density(candidate_point, bad_points)
        ratio = l_value / g_value

        if ratio > best_ratio:
            best_ratio = ratio
            best_candidate = {
                "log10_alpha": float(candidate_point[0]),
                "log10_eta0": float(candidate_point[1]),
            }

    return best_candidate


def run_bohb_like_search(x_train, y_train, x_val, y_val, max_resource=27, eta=3, seed=42):
    """
    教学型简化版 BOHB。

    它把两部分拼在一起：
    - Bayesian Optimization：用历史信息偏向更有希望的参数区域
    - Hyperband：用多预算 / successive halving 节省资源
    """
    import math
    import numpy as np

    rng = np.random.default_rng(seed)
    s_max = int(math.log(max_resource, eta))
    total_budget = (s_max + 1) * max_resource

    history = []
    best_result = None

    for s in range(s_max, -1, -1):
        n = int(math.ceil(total_budget / max_resource * (eta**s) / (s + 1)))
        r = max_resource * (eta ** (-s))
        candidates = [suggest_config_from_history(history, rng) for _ in range(n)]

        print(f"\n=== BOHB Bracket s={s} | 初始候选数={len(candidates)} | 初始资源={r:.2f} ===")

        for i in range(s + 1):
            n_i = max(1, int(n * (eta ** (-i))))
            r_i = max(1, int(r * (eta**i)))

            scored_candidates = []
            for candidate in candidates:
                score = evaluate_config(candidate, r_i, x_train, y_train, x_val, y_val)
                result = {
                    "config": candidate,
                    "score": score,
                    "resource": r_i,
                    "bracket": s,
                    "round": i,
                }
                history.append(result)
                scored_candidates.append(result)

                print(
                    f"轮次={i} | 资源={r_i:2d} | 参数={candidate} "
                    f"-> 验证集准确率={score:.4f}"
                )

                if best_result is None or score > best_result["score"]:
                    best_result = result

            scored_candidates.sort(key=lambda item: item["score"], reverse=True)
            keep_num = max(1, int(math.ceil(n_i / eta)))

            if i < s:
                candidates = [item["config"] for item in scored_candidates[:keep_num]]
                print(f"保留前 {len(candidates)} 个候选者进入下一轮。")

    return best_result, history


def train_final_model(best_config, x_train, y_train, x_val, y_val, x_test, y_test, max_resource):
    """用找到的最好配置做最终测试。"""
    import numpy as np
    from sklearn.linear_model import SGDClassifier
    from sklearn.metrics import accuracy_score

    x_final_train = np.vstack([x_train, x_val])
    y_final_train = np.hstack([y_train, y_val])

    model = SGDClassifier(
        loss="log_loss",
        alpha=10 ** float(best_config["log10_alpha"]),
        learning_rate="constant",
        eta0=10 ** float(best_config["log10_eta0"]),
        max_iter=max_resource,
        tol=None,
        random_state=42,
    )
    model.fit(x_final_train, y_final_train)
    y_test_pred = model.predict(x_test)
    return accuracy_score(y_test, y_test_pred)


def main():
    # 依赖说明：
    # - numpy：采样与核密度计算
    # - scikit-learn：数据集、模型、指标
    try:
        import numpy  # noqa: F401
        import sklearn  # noqa: F401
    except ImportError:
        print("缺少依赖，请安装：pip install numpy scikit-learn")
        return

    max_resource = 27
    x_train, x_val, x_test, y_train, y_val, y_test = prepare_dataset()

    best_result, history = run_bohb_like_search(
        x_train,
        y_train,
        x_val,
        y_val,
        max_resource=max_resource,
        eta=3,
        seed=42,
    )

    test_acc = train_final_model(
        best_result["config"],
        x_train,
        y_train,
        x_val,
        y_val,
        x_test,
        y_test,
        max_resource=max_resource,
    )

    print("\n=== 简化版 BOHB ===")
    print("说明：这里不是工业级 BOHB，而是用 TPE 风格采样 + Hyperband 预算分配来演示组合思想。")
    print(f"历史评估次数：{len(history)}")
    print(f"最佳参数：{best_result['config']}")
    print(f"最佳验证集准确率：{best_result['score']:.4f}")
    print(f"测试集准确率：{test_acc:.4f}")


if __name__ == "__main__":
    main()
