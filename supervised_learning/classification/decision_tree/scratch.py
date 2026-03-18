"""决策树（手写版，信息增益，二分类）。"""

import math


class TreeNode:
    """树节点类。"""

    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        # 内部节点需要 feature_index 和 threshold。
        self.feature_index = feature_index
        self.threshold = threshold

        # 左右子树。
        self.left = left
        self.right = right

        # 叶子节点直接保存类别 value（0 或 1）。
        self.value = value


class DecisionTreeBinaryScratch:
    """使用信息增益训练的二分类决策树。"""

    def __init__(self, max_depth=3, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def fit(self, x_train, y_train):
        """训练入口。"""
        unique_labels = set(y_train)
        if not unique_labels.issubset({0, 1}):
            raise ValueError("当前实现仅支持二分类标签：0 和 1。")
        self.root = self._build_tree(x_train, y_train, depth=0)

    def _entropy(self, labels):
        """计算信息熵。"""
        total = len(labels)
        if total == 0:
            return 0.0

        count0 = sum(1 for label in labels if label == 0)
        count1 = total - count0

        entropy = 0.0
        for count in [count0, count1]:
            if count == 0:
                continue
            prob = count / total
            entropy -= prob * math.log2(prob)
        return entropy

    def _split_dataset(self, x_data, y_data, feature_index, threshold):
        """按 feature_index 和 threshold 切分数据。"""
        left_x, left_y = [], []
        right_x, right_y = [], []

        for features, label in zip(x_data, y_data):
            if features[feature_index] <= threshold:
                left_x.append(features)
                left_y.append(label)
            else:
                right_x.append(features)
                right_y.append(label)

        return left_x, left_y, right_x, right_y

    def _information_gain(self, parent_y, left_y, right_y):
        """计算信息增益：IG = H(parent) - weighted_child_entropy。"""
        parent_entropy = self._entropy(parent_y)
        n = len(parent_y)
        left_weight = len(left_y) / n
        right_weight = len(right_y) / n

        child_entropy = left_weight * self._entropy(left_y) + right_weight * self._entropy(right_y)
        return parent_entropy - child_entropy

    def _best_split(self, x_data, y_data):
        """搜索信息增益最大的特征和阈值。"""
        best_gain = -1.0
        best_feature = None
        best_threshold = None

        n_features = len(x_data[0])
        for feature_index in range(n_features):
            # 候选阈值使用相邻特征值中点，更符合决策边界直觉。
            values = sorted(set(row[feature_index] for row in x_data))
            if len(values) == 1:
                continue
            candidate_thresholds = [
                (values[i] + values[i + 1]) / 2.0 for i in range(len(values) - 1)
            ]

            for threshold in candidate_thresholds:
                left_x, left_y, right_x, right_y = self._split_dataset(
                    x_data, y_data, feature_index, threshold
                )

                # 如果分裂后某边为空，说明这个阈值无效。
                if len(left_y) == 0 or len(right_y) == 0:
                    continue

                gain = self._information_gain(y_data, left_y, right_y)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_index
                    best_threshold = threshold

        return best_feature, best_threshold, best_gain

    def _majority_class(self, labels):
        """返回多数类。"""
        count0 = sum(1 for label in labels if label == 0)
        count1 = len(labels) - count0
        return 1 if count1 > count0 else 0

    def _build_tree(self, x_data, y_data, depth):
        """递归建树。"""
        # 停止条件 1：标签全相同，直接生成叶子。
        if len(set(y_data)) == 1:
            return TreeNode(value=y_data[0])

        # 停止条件 2：达到最大深度。
        if depth >= self.max_depth:
            return TreeNode(value=self._majority_class(y_data))

        # 停止条件 3：样本过少，不再继续分裂。
        if len(y_data) < self.min_samples_split:
            return TreeNode(value=self._majority_class(y_data))

        # 寻找当前最优分裂。
        feature_index, threshold, gain = self._best_split(x_data, y_data)

        # 如果找不到有效分裂或增益太小，则转为叶子。
        if feature_index is None or gain <= 1e-12:
            return TreeNode(value=self._majority_class(y_data))

        # 根据最优分裂切分数据并递归构建子树。
        left_x, left_y, right_x, right_y = self._split_dataset(
            x_data, y_data, feature_index, threshold
        )
        left_node = self._build_tree(left_x, left_y, depth + 1)
        right_node = self._build_tree(right_x, right_y, depth + 1)

        return TreeNode(
            feature_index=feature_index,
            threshold=threshold,
            left=left_node,
            right=right_node,
        )

    def _predict_one(self, features, node):
        """递归预测单个样本。"""
        if node.value is not None:
            return node.value

        if features[node.feature_index] <= node.threshold:
            return self._predict_one(features, node.left)
        return self._predict_one(features, node.right)

    def predict(self, x_test):
        """预测多个样本。"""
        return [self._predict_one(features, self.root) for features in x_test]

    def score(self, x_test, y_test):
        """计算准确率。"""
        y_pred = self.predict(x_test)
        correct = 0
        for true_label, pred_label in zip(y_test, y_pred):
            if true_label == pred_label:
                correct += 1
        return correct / len(y_test)


def main():
    # 1. 构造二分类训练集（标签只有 0/1）。
    x_train = [
        [2.0, 1.0], [2.5, 1.8], [3.0, 1.6], [3.5, 2.0],
        [6.0, 5.0], [6.5, 5.8], [7.0, 6.2], [7.5, 6.8],
        [4.8, 4.2], [5.2, 4.8],
    ]
    y_train = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]

    # 2. 构造测试集。
    x_test = [[2.2, 1.4], [6.8, 5.9], [4.0, 3.0], [5.6, 5.0]]
    y_test = [0, 1, 0, 1]

    # 3. 创建并训练决策树模型。
    model = DecisionTreeBinaryScratch(max_depth=3, min_samples_split=2)
    model.fit(x_train, y_train)

    # 4. 预测并评估。
    y_pred = model.predict(x_test)
    acc = model.score(x_test, y_test)

    print("=== 决策树（手写版，信息增益）===")
    print("预测结果：", y_pred)
    print(f"测试集准确率：{acc:.4f}")


if __name__ == "__main__":
    main()
