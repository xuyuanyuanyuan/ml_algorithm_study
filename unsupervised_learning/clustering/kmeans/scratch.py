"""K-Means 聚类（手写版）。"""

import math
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


class KMeansScratch:
    """从零实现的 KMeans 聚类器。"""

    def __init__(self, n_clusters=3, max_iter=100, tol=1e-4, random_state=42):
        # n_clusters：簇数量；max_iter：最大迭代轮数；tol：收敛阈值。
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

        # 训练后保存的结果。
        self.centroids_ = []
        self.labels_ = []
        self.n_iter_ = 0

    def _euclidean_distance(self, point_a, point_b):
        """计算欧氏距离。"""
        total = 0.0
        for a, b in zip(point_a, point_b):
            total += (a - b) ** 2
        return math.sqrt(total)

    def _initialize_centroids(self, x_data):
        """随机初始化聚类中心（从样本中随机选点）。"""
        random.seed(self.random_state)
        self.centroids_ = random.sample(x_data, self.n_clusters)

    def _assign_clusters(self, x_data):
        """把每个样本分配到最近的聚类中心。"""
        labels = []
        for sample in x_data:
            distances = [
                self._euclidean_distance(sample, center) for center in self.centroids_
            ]
            label = distances.index(min(distances))
            labels.append(label)
        return labels

    def _mean_point(self, points):
        """计算一个簇内样本的均值点。"""
        dims = len(points[0])
        mean = []
        for dim in range(dims):
            dim_sum = sum(point[dim] for point in points)
            mean.append(dim_sum / len(points))
        return mean

    def _update_centroids(self, x_data, labels, old_centroids):
        """根据当前标签更新每个簇中心。"""
        new_centroids = []
        for cluster_id in range(self.n_clusters):
            cluster_points = [
                point for point, label in zip(x_data, labels) if label == cluster_id
            ]

            if cluster_points:
                new_centroids.append(self._mean_point(cluster_points))
            else:
                # 如果某个簇为空，保留旧中心。
                new_centroids.append(old_centroids[cluster_id])
        return new_centroids

    def _centroid_shift(self, old_centroids, new_centroids):
        """计算所有中心总位移，用于判断是否收敛。"""
        shift = 0.0
        for old, new in zip(old_centroids, new_centroids):
            shift += self._euclidean_distance(old, new)
        return shift

    def fit(self, x_data):
        """训练 KMeans：分配样本 -> 更新中心 -> 判断收敛。"""
        self._initialize_centroids(x_data)

        for epoch in range(1, self.max_iter + 1):
            # 1. 根据当前中心分配簇标签。
            labels = self._assign_clusters(x_data)

            # 2. 计算新的中心。
            old_centroids = self.centroids_
            new_centroids = self._update_centroids(x_data, labels, old_centroids)

            # 3. 判断中心位移是否小于阈值。
            shift = self._centroid_shift(old_centroids, new_centroids)
            self.centroids_ = new_centroids
            self.labels_ = labels
            self.n_iter_ = epoch

            if shift < self.tol:
                break

    def predict(self, x_data):
        """对新样本分配簇标签。"""
        labels = []
        for sample in x_data:
            distances = [
                self._euclidean_distance(sample, center) for center in self.centroids_
            ]
            labels.append(distances.index(min(distances)))
        return labels


def build_sample_data():
    """手工生成 3 簇二维样本。"""
    random.seed(7)
    data = []

    # 以三个中心点为基础，加入高斯噪声。
    centers = [(1.0, 1.0), (5.0, 5.0), (8.0, 1.5)]
    for cx, cy in centers:
        for _ in range(35):
            x = random.gauss(cx, 0.45)
            y = random.gauss(cy, 0.45)
            data.append([x, y])
    return data


def main():
    # 1. 准备二维聚类数据。
    x_data = build_sample_data()

    # 2. 创建并训练 KMeans 模型。
    model = KMeansScratch(n_clusters=3, max_iter=100, tol=1e-4, random_state=42)
    model.fit(x_data)

    # 3. 输出结果。
    print("=== KMeans（手写版）===")
    print(f"迭代轮数：{model.n_iter_}")
    for idx, center in enumerate(model.centroids_):
        count = sum(1 for label in model.labels_ if label == idx)
        print(f"簇 {idx} 中心：{center}，样本数：{count}")

    # 4. 绘制聚类结果。
    plt = _get_styled_pyplot()
    if plt is None:
        print("未安装 matplotlib，跳过绘图。可安装：pip install matplotlib")
        return

    x_axis = [point[0] for point in x_data]
    y_axis = [point[1] for point in x_data]

    plt.figure()
    plt.scatter(x_axis, y_axis, c=model.labels_, cmap="tab10", s=38, alpha=0.8)
    plt.scatter(
        [c[0] for c in model.centroids_],
        [c[1] for c in model.centroids_],
        c="tab:red",
        marker="X",
        s=220,
        label="聚类中心",
    )
    plt.title("KMeans 手写版聚类结果")
    plt.xlabel("特征 1")
    plt.ylabel("特征 2")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
