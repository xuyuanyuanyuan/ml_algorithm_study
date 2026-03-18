"""线性回归（scikit-learn 版）：自动造数据、训练、评估并绘图。"""


def _get_styled_pyplot():
    """获取统一风格的 pyplot。"""
    try:
        import sys
        from pathlib import Path

        project_root = None
        for parent in Path(__file__).resolve().parents:
            if (parent / "utils" / "plot_utils.py").exists():
                project_root = parent
                break
        if project_root is None:
            return None
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        from utils.plot_utils import get_styled_pyplot
    except Exception:
        return None
    return get_styled_pyplot()


def main():
    # 1. 导入依赖库。
    try:
        import numpy as np
        from sklearn.datasets import make_regression
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import mean_squared_error
        from sklearn.model_selection import train_test_split
    except ImportError:
        print("未检测到 scikit-learn。请先安装：pip install scikit-learn")
        return

    # 2. 自动生成一元回归数据（只有一个特征，便于画图）。
    X, y = make_regression(
        n_samples=120,
        n_features=1,
        noise=15.0,
        random_state=42,
    )

    # 3. 划分训练集和测试集。
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    # 4. 创建并训练线性回归模型。
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 5. 在测试集上预测并计算 MSE。
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    # 6. 输出关键结果。
    print("=== 线性回归（sklearn）===")
    print(f"模型系数（斜率）：{model.coef_[0]:.4f}")
    print(f"模型截距：{model.intercept_:.4f}")
    print(f"测试集 MSE：{mse:.4f}")

    # 7. 使用 matplotlib 画出散点和回归直线。
    plt = _get_styled_pyplot()
    if plt is None:
        print("未检测到 matplotlib，跳过绘图。可安装：pip install matplotlib")
        return

    # 用均匀的 x 取值生成一条直线。
    x_line = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)
    y_line = model.predict(x_line)

    plt.figure()
    plt.scatter(X[:, 0], y, color="tab:blue", alpha=0.65, label="样本点")
    plt.plot(x_line[:, 0], y_line, color="tab:orange", label="回归直线")
    plt.title("LinearRegression 回归效果图")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
