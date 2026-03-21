"""高斯过程贝叶斯优化（skopt 版）：使用 surrogate model + acquisition function 搜索超参数。"""


def main():
    # 依赖说明：
    # - scikit-learn：数据集、模型、交叉验证
    # - scikit-optimize：gp_minimize
    try:
        from skopt import gp_minimize
        from skopt.space import Real
        from skopt.utils import use_named_args
        from sklearn.datasets import load_iris
        from sklearn.model_selection import cross_val_score
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC
    except ImportError:
        print("缺少依赖，请安装：pip install scikit-learn scikit-optimize")
        print("说明：scikit-optimize 简称 skopt，gp_minimize 会用高斯过程近似目标函数。")
        return

    data = load_iris()

    # 搜索空间：
    # - C：软间隔惩罚系数
    # - gamma：RBF 核宽度
    search_space = [
        Real(1e-2, 100.0, prior="log-uniform", name="C"),
        Real(1e-3, 1.0, prior="log-uniform", name="gamma"),
    ]

    @use_named_args(search_space)
    def objective(C, gamma):
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("svc", SVC(kernel="rbf", C=C, gamma=gamma)),
            ]
        )
        score = cross_val_score(model, data.data, data.target, cv=5, scoring="accuracy").mean()
        # skopt 默认做最小化，所以这里返回 1 - accuracy。
        return 1.0 - float(score)

    result = gp_minimize(
        func=objective,
        dimensions=search_space,
        n_calls=18,
        n_initial_points=6,
        random_state=42,
    )

    print("=== GP Bayesian Optimization Demo ===")
    print(f"最优参数：C={result.x[0]:.6f}, gamma={result.x[1]:.6f}")
    print(f"最佳交叉验证准确率：{1.0 - result.fun:.4f}")
    print(f"总试验次数：{len(result.func_vals)}")


if __name__ == "__main__":
    main()
