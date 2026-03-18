"""FP-Growth（手写教学模板版）: 用简化频繁模式树思想示意。"""


class FPGrowthScratchTemplate:
    """
    教学模板:
    - 正式 FP-Growth 会建 FP-Tree + 条件模式基递归挖掘
    - 这里用简化统计展示“高频前缀”思路，便于初学者理解
    """

    def __init__(self, min_support=0.3):
        self.min_support = min_support

    def fit(self, transactions):
        from collections import Counter

        n = len(transactions)
        # 1. 统计 1 项频率。
        c1 = Counter(item for t in transactions for item in t)
        freq1 = {item: cnt / n for item, cnt in c1.items() if cnt / n >= self.min_support}

        # 2. 统计“按频率排序后的前缀二项”频率（简化展示）。
        pairs = Counter()
        for t in transactions:
            filtered = [i for i in t if i in freq1]
            filtered.sort(key=lambda i: c1[i], reverse=True)
            if len(filtered) >= 2:
                pairs[(filtered[0], filtered[1])] += 1
        freq2 = {k: v / n for k, v in pairs.items() if v / n >= self.min_support}

        self.freq1_ = freq1
        self.freq2_ = freq2
        return freq1, freq2


def main():
    transactions = [
        ["啤酒", "尿布", "面包"],
        ["啤酒", "尿布", "牛奶"],
        ["尿布", "面包"],
        ["啤酒", "面包"],
        ["牛奶", "面包"],
        ["啤酒", "尿布", "面包", "牛奶"],
    ]

    model = FPGrowthScratchTemplate(min_support=0.3)
    freq1, freq2 = model.fit(transactions)
    print("=== FP-Growth Scratch Template ===")
    print("频繁1项:", freq1)
    print("高频前缀二项(简化):", freq2)


if __name__ == "__main__":
    main()
