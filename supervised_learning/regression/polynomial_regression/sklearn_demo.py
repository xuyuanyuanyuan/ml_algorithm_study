"""多项式回归（sklearn 版）: 用非线性特征 + 线性回归拟合曲线。"""


def main():
    # 关键参数: degree 表示多项式阶数，越大模型越灵活。
    degree = 3

    try:
        import numpy as np
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import mean_squared_error
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import PolynomialFeatures
    except ImportError:
        print("缺少依赖，请安装: pip install scikit-learn numpy")
        return

    # 1. 构造带噪声的非线性数据。
    rng = np.random.default_rng(42)
    x = np.linspace(-3, 3, 120).reshape(-1, 1)
    y = 0.8 * x[:, 0] ** 3 - 1.2 * x[:, 0] ** 2 + 0.5 * x[:, 0] + rng.normal(0, 2, 120)

    # 2. 建立“多项式特征 + 线性回归”管道。
    model = Pipeline(
        [
            ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
            ("lr", LinearRegression()),
        ]
    )
    model.fit(x, y)
    pred = model.predict(x)
    mse = mean_squared_error(y, pred)

    print("=== Polynomial Regression (sklearn) ===")
    print(f"degree={degree}, MSE={mse:.4f}")


if __name__ == "__main__":
    main()
