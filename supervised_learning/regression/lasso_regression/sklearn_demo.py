"""Lasso 回归（sklearn 版）: 使用 L1 正则做特征选择。"""


def main():
    # 关键参数: alpha 是 L1 正则强度，越大越稀疏。
    alpha = 0.1

    try:
        from sklearn.datasets import make_regression
        from sklearn.linear_model import Lasso
        from sklearn.metrics import mean_squared_error
        from sklearn.model_selection import train_test_split
    except ImportError:
        print("缺少依赖，请安装: pip install scikit-learn")
        return

    x, y = make_regression(
        n_samples=200, n_features=20, n_informative=5, noise=10, random_state=42
    )
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    model = Lasso(alpha=alpha, max_iter=5000, random_state=42)
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    mse = mean_squared_error(y_test, pred)
    non_zero = int((model.coef_ != 0).sum())

    print("=== Lasso Regression (sklearn) ===")
    print(f"alpha={alpha}, MSE={mse:.4f}, 非零系数个数={non_zero}")


if __name__ == "__main__":
    main()
