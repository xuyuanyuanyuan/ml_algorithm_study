"""DBSCAN（手写简化版）: 核心点扩展聚类。"""


class DBSCANScratch:
    """教学版 DBSCAN。"""

    def __init__(self, eps=0.25, min_samples=4):
        self.eps = eps
        self.min_samples = min_samples

    def _dist(self, a, b):
        import math

        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

    def _region_query(self, x_data, idx):
        neighbors = []
        for j, sample in enumerate(x_data):
            if self._dist(x_data[idx], sample) <= self.eps:
                neighbors.append(j)
        return neighbors

    def fit_predict(self, x_data):
        labels = [0] * len(x_data)  # 0: 未访问, -1: 噪声, >0: 簇编号
        cluster_id = 0

        for i in range(len(x_data)):
            if labels[i] != 0:
                continue
            neighbors = self._region_query(x_data, i)
            if len(neighbors) < self.min_samples:
                labels[i] = -1
                continue

            cluster_id += 1
            labels[i] = cluster_id
            queue = neighbors[:]

            while queue:
                j = queue.pop(0)
                if labels[j] == -1:
                    labels[j] = cluster_id
                if labels[j] != 0:
                    continue
                labels[j] = cluster_id
                j_neighbors = self._region_query(x_data, j)
                if len(j_neighbors) >= self.min_samples:
                    queue.extend(j_neighbors)

        return labels


def main():
    import random

    random.seed(42)
    x_data = []
    for _ in range(70):
        x_data.append([random.gauss(0, 0.25), random.gauss(0, 0.25)])
        x_data.append([random.gauss(2, 0.25), random.gauss(2, 0.25)])
    x_data.extend([[4.5, 4.5], [4.8, 4.7], [-1.5, 2.8]])

    model = DBSCANScratch(eps=0.35, min_samples=5)
    labels = model.fit_predict(x_data)
    noise_count = sum(1 for v in labels if v == -1)
    print("=== DBSCAN (scratch) ===")
    print(f"噪声点数量={noise_count}, 标签前10个={labels[:10]}")


if __name__ == "__main__":
    main()
