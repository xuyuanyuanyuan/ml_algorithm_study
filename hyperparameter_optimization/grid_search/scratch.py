"""网格搜索（手写最小版）：穷举参数组合并比较验证集表现。"""


def generate_param_combinations(param_grid):
    """把参数网格展开成一个个具体组合。"""
    from itertools import product

    keys = list(param_grid.keys())
    value_lists = [param_grid[key] for key in keys]

    for values in product(*value_lists):
        yield dict(zip(keys, values))


def build_model(params):
    """根据一组参数构建模型。"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=params["C"],
                    max_iter=params["max_iter"],
                    solver="lbfgs",
                    random_state=42,
                ),
            ),
        ]
    )


def manual_grid_search(x_train, y_train, x_val, y_val, param_grid):
    """
    手写最小网格搜索流程：
    - 给定参数网格
    - 穷举所有组合
    - 在训练集训练
    - 在验证集比较表现
    - 返回最优参数
    """
    from sklearn.metrics import accuracy_score

    best_params = None
    best_score = -1.0

    for index, params in enumerate(generate_param_combinations(param_grid), start=1):
        model = build_model(params)
        model.fit(x_train, y_train)
        y_val_pred = model.predict(x_val)
        val_score = accuracy_score(y_val, y_val_pred)

        print(f"第 {index:2d} 组参数：{params} -> 验证集准确率={val_score:.4f}")

        if val_score > best_score:
            best_score = val_score
            best_params = params

    return best_params, best_score


def main():
    # 依赖说明：
    # - numpy：数组拼接
    # - scikit-learn：数据集、模型、指标
    try:
        import numpy as np
        from sklearn.datasets import load_breast_cancer
        from sklearn.metrics import accuracy_score
        from sklearn.model_selection import train_test_split
    except ImportError:
        print("缺少依赖，请安装：pip install numpy scikit-learn")
        return

    # 1. 准备数据，并拆成训练集 / 验证集 / 测试集。
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

    # 2. 定义参数网格。
    param_grid = {
        "C": [0.01, 0.1, 1.0, 10.0],
        "max_iter": [300, 800],
    }

    # 3. 手写网格搜索。
    best_params, best_val_score = manual_grid_search(
        x_train,
        y_train,
        x_val,
        y_val,
        param_grid,
    )

    # 4. 用最优参数在训练集 + 验证集上重新训练，再去测试集评估。
    best_model = build_model(best_params)
    x_final_train = np.vstack([x_train, x_val])
    y_final_train = np.hstack([y_train, y_val])
    best_model.fit(x_final_train, y_final_train)

    y_test_pred = best_model.predict(x_test)
    test_acc = accuracy_score(y_test, y_test_pred)

    print("\n=== 手写 Grid Search ===")
    print(f"最优参数：{best_params}")
    print(f"最佳验证集准确率：{best_val_score:.4f}")
    print(f"测试集准确率：{test_acc:.4f}")


if __name__ == "__main__":
    main()
