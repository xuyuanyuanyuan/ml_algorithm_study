"""SMAC（库版依赖模板）：优先使用 SMAC 库，如接口版本不同请按注释微调。"""


def main():
    # 依赖说明：
    # - scikit-learn：数据集、模型、交叉验证
    # - smac：SMAC 优化器
    # - ConfigSpace：定义搜索空间
    try:
        from ConfigSpace import ConfigurationSpace, Float, Integer
        from smac import HyperparameterOptimizationFacade, Scenario
        from sklearn.datasets import load_breast_cancer
        from sklearn.model_selection import cross_val_score
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC
    except ImportError:
        print("缺少依赖，请安装：pip install smac ConfigSpace scikit-learn")
        print("说明：SMAC 的不同版本接口略有差异，本文件保留为教学型库版模板。")
        return

    data = load_breast_cancer()

    def objective(config, seed=42):
        c_value = 10 ** float(config["log10_C"])
        max_iter = int(config["max_iter"])

        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("svc", SVC(kernel="linear", C=c_value, max_iter=max_iter, random_state=seed)),
            ]
        )
        score = cross_val_score(model, data.data, data.target, cv=5, scoring="accuracy").mean()
        # SMAC 默认做最小化，所以返回 1 - accuracy。
        return 1.0 - float(score)

    try:
        # 这里演示如何定义一个“连续 + 整数”混合搜索空间。
        configspace = ConfigurationSpace(seed=42)
        configspace.add_hyperparameters(
            [
                Float("log10_C", bounds=(-3.0, 2.0)),
                Integer("max_iter", bounds=(300, 1200)),
            ]
        )

        scenario = Scenario(
            configspace=configspace,
            deterministic=True,
            n_trials=15,
            seed=42,
        )

        optimizer = HyperparameterOptimizationFacade(
            scenario=scenario,
            target_function=objective,
        )
        incumbent = optimizer.optimize()
        best_score = 1.0 - objective(incumbent)

        print("=== SMAC Demo ===")
        print(f"最优参数：log10_C={float(incumbent['log10_C']):.4f}, max_iter={int(incumbent['max_iter'])}")
        print(f"最优 C：{10 ** float(incumbent['log10_C']):.6f}")
        print(f"最佳交叉验证准确率：{best_score:.4f}")
    except Exception as exc:
        print("SMAC 已安装，但当前环境中的接口版本与本模板可能不同。")
        print(f"具体异常：{exc}")
        print("你可以保留本文件作为教学模板，再根据本地 smac 版本微调导入方式或 Scenario 参数。")


if __name__ == "__main__":
    main()
