"""Hyperband（手写教学简化版）：展示多组参数如何通过 successive halving 逐步筛选。"""


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


def sample_candidate(rng):
    """随机采样一组超参数。"""
    return {
        "alpha": 10 ** rng.uniform(-5.0, -2.0),
        "eta0": 10 ** rng.uniform(-3.0, -0.5),
    }


def evaluate_candidate(candidate, resource, x_train, y_train, x_val, y_val):
    """
    用给定资源训练模型，并在验证集上打分。

    这里把 SGDClassifier 的 max_iter 当作资源：
    - 资源小：训练轮数少，先做快速筛选
    - 资源大：对保留下来的候选者投入更多训练成本
    """
    from sklearn.linear_model import SGDClassifier
    from sklearn.metrics import accuracy_score

    model = SGDClassifier(
        loss="log_loss",
        alpha=candidate["alpha"],
        learning_rate="constant",
        eta0=candidate["eta0"],
        max_iter=max(1, int(resource)),
        tol=None,
        random_state=42,
    )
    model.fit(x_train, y_train)
    y_val_pred = model.predict(x_val)
    return accuracy_score(y_val, y_val_pred)


def run_hyperband(x_train, y_train, x_val, y_val, max_resource=27, eta=3, seed=42):
    """
    手写最小 Hyperband：
    - 外层：不同 bracket
    - 内层：successive halving
    - 先给很多参数较少资源，再逐轮淘汰
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
        candidates = [sample_candidate(rng) for _ in range(n)]

        print(f"\n=== Bracket s={s} | 初始候选数={len(candidates)} | 初始资源={r:.2f} ===")

        for i in range(s + 1):
            n_i = max(1, int(n * (eta ** (-i))))
            r_i = max(1, int(r * (eta**i)))

            scored_candidates = []
            for candidate in candidates:
                score = evaluate_candidate(candidate, r_i, x_train, y_train, x_val, y_val)
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

            keep_num = max(1, int(np.ceil(n_i / eta)))
            if i < s:
                candidates = [item["config"] for item in scored_candidates[:keep_num]]
                print(f"保留前 {len(candidates)} 个候选者进入下一轮。")

    return best_result, history


def train_final_model(best_config, x_train, y_train, x_val, y_val, x_test, y_test, max_resource):
    """用找到的最好超参数，在更大训练数据上做最终评估。"""
    import numpy as np
    from sklearn.linear_model import SGDClassifier
    from sklearn.metrics import accuracy_score

    x_final_train = np.vstack([x_train, x_val])
    y_final_train = np.hstack([y_train, y_val])

    model = SGDClassifier(
        loss="log_loss",
        alpha=best_config["alpha"],
        learning_rate="constant",
        eta0=best_config["eta0"],
        max_iter=max_resource,
        tol=None,
        random_state=42,
    )
    model.fit(x_final_train, y_final_train)
    y_test_pred = model.predict(x_test)
    return accuracy_score(y_test, y_test_pred)


def main():
    # 依赖说明：
    # - numpy：数学与采样
    # - scikit-learn：数据集、模型、指标
    try:
        import numpy  # noqa: F401
        import sklearn  # noqa: F401
    except ImportError:
        print("缺少依赖，请安装：pip install numpy scikit-learn")
        return

    max_resource = 27

    # 1. 准备数据。
    x_train, x_val, x_test, y_train, y_val, y_test = prepare_dataset()

    # 2. 运行手写 Hyperband。
    best_result, _history = run_hyperband(
        x_train,
        y_train,
        x_val,
        y_val,
        max_resource=max_resource,
        eta=3,
        seed=42,
    )

    # 3. 用最好参数做最终测试。
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

    print("\n=== 手写 Hyperband ===")
    print(f"最佳参数：{best_result['config']}")
    print(f"最佳验证集准确率：{best_result['score']:.4f}")
    print(f"对应资源：{best_result['resource']}")
    print(f"测试集准确率：{test_acc:.4f}")


if __name__ == "__main__":
    main()
