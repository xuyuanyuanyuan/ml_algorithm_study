"""线性回归（手写梯度下降版）。"""

import random


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


class LinearRegressionScratch:
    """从零实现的一元线性回归模型。"""

    def __init__(self, learning_rate=0.01, epochs=1000):
        # 学习率控制每一步更新幅度，epochs 表示训练轮数。
        self.learning_rate = learning_rate
        self.epochs = epochs

        # 参数初始化：w 和 b 初始为 0。
        self.weight = 0.0
        self.bias = 0.0

        # 记录每轮损失，便于观察训练过程。
        self.loss_history = []

    def _initialize_parameters(self):
        """参数初始化函数。"""
        self.weight = 0.0
        self.bias = 0.0

    def predict_one(self, x):
        """前向预测（单样本）：y_hat = w*x + b。"""
        return self.weight * x + self.bias

    def predict(self, x_list):
        """前向预测（多样本）。"""
        return [self.predict_one(x) for x in x_list]

    def mse_loss(self, x_list, y_list):
        """计算均方误差（MSE）。"""
        n = len(x_list)
        total_error = 0.0
        for x, y in zip(x_list, y_list):
            error = self.predict_one(x) - y
            total_error += error * error
        return total_error / n

    def compute_gradients(self, x_list, y_list):
        """根据当前参数计算 MSE 对 w、b 的梯度。"""
        n = len(x_list)
        grad_w = 0.0
        grad_b = 0.0

        for x, y in zip(x_list, y_list):
            error = self.predict_one(x) - y
            grad_w += (2 / n) * error * x
            grad_b += (2 / n) * error

        return grad_w, grad_b

    def fit(self, x_list, y_list, print_every=100):
        """训练模型：梯度计算 -> 参数更新 -> 记录损失。"""
        self._initialize_parameters()
        self.loss_history = []

        for epoch in range(1, self.epochs + 1):
            # 1. 梯度计算。
            grad_w, grad_b = self.compute_gradients(x_list, y_list)

            # 2. 参数更新（梯度下降）。
            self.weight -= self.learning_rate * grad_w
            self.bias -= self.learning_rate * grad_b

            # 3. 记录当前损失。
            loss = self.mse_loss(x_list, y_list)
            self.loss_history.append(loss)

            # 打印训练信息，方便初学者观察。
            if epoch % print_every == 0:
                print(f"第 {epoch:4d} 轮，MSE={loss:.6f}")


def build_sample_data():
    """构造近似 y = 3x + 5 的训练数据。"""
    random.seed(42)
    x_list = []
    y_list = []
    for x in range(1, 31):
        noise = random.uniform(-2.0, 2.0)
        y = 3 * x + 5 + noise
        x_list.append(float(x))
        y_list.append(y)
    return x_list, y_list


def main():
    # 1. 准备数据。
    x_train, y_train = build_sample_data()

    # 2. 创建并训练模型。
    model = LinearRegressionScratch(learning_rate=0.002, epochs=3000)
    model.fit(x_train, y_train, print_every=500)

    # 3. 打印训练结果。
    final_loss = model.mse_loss(x_train, y_train)
    print("=== 线性回归（手写版）===")
    print(f"学习到的 weight（斜率）: {model.weight:.4f}")
    print(f"学习到的 bias（截距）: {model.bias:.4f}")
    print(f"最终 MSE: {final_loss:.6f}")

    # 4. 预测演示。
    test_x = 35.0
    pred_y = model.predict_one(test_x)
    print(f"x={test_x:.1f} 时，预测值 y={pred_y:.4f}")

    # 5. 绘制训练结果（散点 + 回归线 + 损失曲线）。
    plt = _get_styled_pyplot()
    if plt is None:
        print("未安装 matplotlib，跳过绘图。可安装：pip install matplotlib")
        return

    # 回归线：取训练数据区间内的点来画直线。
    x_line = [
        min(x_train) + i * 0.1
        for i in range(int((max(x_train) - min(x_train)) * 10) + 1)
    ]
    y_line = model.predict(x_line)

    plt.figure(figsize=(10, 4.2))

    plt.subplot(1, 2, 1)
    plt.scatter(x_train, y_train, color="tab:blue", alpha=0.7, label="训练样本")
    plt.plot(x_line, y_line, color="tab:orange", label="回归线")
    plt.title("线性回归拟合结果")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(model.loss_history, color="tab:green")
    plt.title("训练损失曲线（MSE）")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
