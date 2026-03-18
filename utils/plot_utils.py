"""绘图工具函数（统一 matplotlib 风格，适合教学演示）。"""


def apply_plot_style(plt):
    """统一项目中的 matplotlib 风格。"""
    try:
        # 该样式在新版 matplotlib 中更稳定。
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        # 兼容旧版本 matplotlib。
        plt.style.use("ggplot")

    plt.rcParams.update(
        {
            "figure.figsize": (8, 5),
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "axes.unicode_minus": False,
            "font.sans-serif": [
                "Microsoft YaHei",
                "SimHei",
                "Arial Unicode MS",
                "DejaVu Sans",
            ],
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "lines.linewidth": 2.0,
            "legend.frameon": True,
            "legend.framealpha": 0.9,
        }
    )


def get_styled_pyplot():
    """获取应用了统一样式的 pyplot 对象。"""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    apply_plot_style(plt)
    return plt


def plot_2d_points(points, labels=None, title="二维散点图"):
    """
    绘制二维点。
    - points: 形如 [[x1, y1], [x2, y2], ...]
    - labels: 可选，类别标签列表
    """
    plt = get_styled_pyplot()
    if plt is None:
        print("未检测到 matplotlib。请先执行：pip install matplotlib")
        return

    x_values = [p[0] for p in points]
    y_values = [p[1] for p in points]

    plt.figure()
    if labels is None:
        plt.scatter(x_values, y_values, c="tab:blue", s=45)
    else:
        plt.scatter(x_values, y_values, c=labels, cmap="tab10", s=45)

    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.tight_layout()
    plt.show()


def main():
    # 使用一组简单点测试绘图函数。
    sample_points = [[1, 1], [1.5, 2], [2, 1.2], [5, 5], [5.5, 4.5], [6, 5.2]]
    sample_labels = [0, 0, 0, 1, 1, 1]
    plot_2d_points(sample_points, sample_labels, title="plot_utils 测试图")


if __name__ == "__main__":
    main()
