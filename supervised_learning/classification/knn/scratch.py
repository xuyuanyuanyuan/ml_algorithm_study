"""KNN 分类（手写版）。"""

import math


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


class KNNClassifierScratch:
    """从零实现的 KNN 分类器。"""

    def __init__(self, k=3):
        # k 表示投票时使用的最近邻个数。
        self.k = k
        self.x_train = []
        self.y_train = []

    def fit(self, x_train, y_train):
        """保存训练数据（KNN 属于“懒惰学习”，训练阶段几乎不做计算）。"""
        self.x_train = x_train
        self.y_train = y_train

    def euclidean_distance(self, point_a, point_b):
        """欧氏距离：sqrt((x1-x2)^2 + (y1-y2)^2 + ...)。"""
        total = 0.0
        for a, b in zip(point_a, point_b):
            total += (a - b) ** 2
        return math.sqrt(total)

    def get_k_neighbors(self, sample):
        """找到与 sample 距离最近的 k 个训练样本。"""
        distance_and_label = []
        for x_item, y_item in zip(self.x_train, self.y_train):
            dist = self.euclidean_distance(x_item, sample)
            distance_and_label.append((dist, y_item))

        # 按距离升序排序后取前 k 个。
        distance_and_label.sort(key=lambda item: item[0])
        return distance_and_label[: self.k]

    def majority_vote(self, neighbor_items):
        """多数投票：统计 k 个邻居中哪个类别出现次数最多。"""
        vote_count = {}
        for _, label in neighbor_items:
            vote_count[label] = vote_count.get(label, 0) + 1

        # 如果票数相同，max 会按字典顺序返回第一个最大值。
        return max(vote_count.items(), key=lambda item: item[1])[0]

    def predict_one(self, sample):
        """预测单个样本类别。"""
        neighbors = self.get_k_neighbors(sample)
        return self.majority_vote(neighbors)

    def predict(self, x_test):
        """预测多个样本类别。"""
        return [self.predict_one(sample) for sample in x_test]

    def score(self, x_test, y_test):
        """计算准确率。"""
        pred_y = self.predict(x_test)
        correct = 0
        for true_label, pred_label in zip(y_test, pred_y):
            if true_label == pred_label:
                correct += 1
        return correct / len(y_test)


def main():
    # 1. 构造简单二维训练数据（两类）。
    x_train = [
        [1.0, 1.0],
        [1.4, 1.6],
        [2.0, 1.8],
        [5.8, 6.2],
        [6.5, 5.9],
        [7.2, 6.8],
    ]
    y_train = [0, 0, 0, 1, 1, 1]

    # 2. 构造测试数据。
    x_test = [
        [1.5, 1.2],
        [6.8, 6.0],
        [3.6, 3.4],
    ]
    y_test = [0, 1, 0]

    # 3. 创建并“训练”KNN 模型。
    model = KNNClassifierScratch(k=3)
    model.fit(x_train, y_train)

    # 4. 预测并输出结果。
    pred_y = model.predict(x_test)
    acc = model.score(x_test, y_test)
    print("=== KNN（手写版）===")
    print("测试集预测结果：", pred_y)
    print(f"测试集准确率：{acc:.4f}")

    # 5. 可视化二维点分布（训练集 + 测试集）。
    plt = _get_styled_pyplot()
    if plt is None:
        print("未安装 matplotlib，跳过绘图。可安装：pip install matplotlib")
        return

    # 按类别拆分训练集，方便用不同颜色显示。
    class0_x = [p[0] for p, y in zip(x_train, y_train) if y == 0]
    class0_y = [p[1] for p, y in zip(x_train, y_train) if y == 0]
    class1_x = [p[0] for p, y in zip(x_train, y_train) if y == 1]
    class1_y = [p[1] for p, y in zip(x_train, y_train) if y == 1]

    plt.figure()
    plt.scatter(class0_x, class0_y, c="tab:blue", s=70, label="训练类 0")
    plt.scatter(class1_x, class1_y, c="tab:orange", s=70, label="训练类 1")
    plt.scatter(
        [p[0] for p in x_test],
        [p[1] for p in x_test],
        c=pred_y,
        cmap="tab10",
        marker="x",
        s=120,
        linewidths=2,
        label="测试点（按预测着色）",
    )
    plt.title("KNN 手写版：二维数据分类结果")
    plt.xlabel("特征 1")
    plt.ylabel("特征 2")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
