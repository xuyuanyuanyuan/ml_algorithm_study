"""层次聚类（手写简化版）: 单链接策略。"""


class HierarchicalClusteringScratch:
    """教学版层次聚类: 每次合并最近两个簇。"""

    def __init__(self, n_clusters=3):
        self.n_clusters = n_clusters

    def _distance(self, a, b):
        import math

        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

    def fit_predict(self, x_data):
        # 初始每个样本是一个簇，簇里保存样本索引。
        clusters = [[i] for i in range(len(x_data))]

        while len(clusters) > self.n_clusters:
            best_i, best_j, best_d = 0, 1, float("inf")
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    # 单链接: 两簇中最近点距离。
                    d = min(
                        self._distance(x_data[p], x_data[q])
                        for p in clusters[i]
                        for q in clusters[j]
                    )
                    if d < best_d:
                        best_i, best_j, best_d = i, j, d
            clusters[best_i] = clusters[best_i] + clusters[best_j]
            clusters.pop(best_j)

        labels = [0] * len(x_data)
        for cid, cluster in enumerate(clusters):
            for idx in cluster:
                labels[idx] = cid
        return labels


def main():
    import random

    random.seed(42)
    x_data = []
    for _ in range(20):
        x_data.append([random.uniform(0, 1), random.uniform(0, 1)])
        x_data.append([random.uniform(3, 4), random.uniform(3, 4)])
        x_data.append([random.uniform(6, 7), random.uniform(0, 1)])

    model = HierarchicalClusteringScratch(n_clusters=3)
    labels = model.fit_predict(x_data)
    print("=== Hierarchical Clustering (scratch) ===")
    print(f"簇标签前10个: {labels[:10]}")


if __name__ == "__main__":
    main()
