"""Apriori（手写简化版）: 频繁 1/2/3 项集 + 基础规则生成。"""


class AprioriScratch:
    """教学版 Apriori。"""

    def __init__(self, min_support=0.3, min_confidence=0.6):
        self.min_support = min_support
        self.min_confidence = min_confidence

    def _support(self, itemset, transactions):
        count = 0
        for t in transactions:
            if itemset.issubset(set(t)):
                count += 1
        return count / len(transactions)

    def fit(self, transactions):
        items = sorted(set(item for t in transactions for item in t))
        freq = {}

        # 1 项集。
        l1 = []
        for item in items:
            s = self._support({item}, transactions)
            if s >= self.min_support:
                l1.append(frozenset([item]))
                freq[frozenset([item])] = s

        # 2 项集。
        l2 = []
        for i in range(len(l1)):
            for j in range(i + 1, len(l1)):
                cand = l1[i] | l1[j]
                s = self._support(set(cand), transactions)
                if s >= self.min_support:
                    l2.append(cand)
                    freq[cand] = s

        self.freq_itemsets = freq
        return freq

    def generate_rules(self):
        rules = []
        for itemset, sup in self.freq_itemsets.items():
            if len(itemset) < 2:
                continue
            for item in itemset:
                lhs = frozenset([item])
                rhs = itemset - lhs
                conf = sup / self.freq_itemsets.get(lhs, 1e-12)
                if conf >= self.min_confidence:
                    rules.append((set(lhs), set(rhs), sup, conf))
        return rules


def main():
    transactions = [
        ["牛奶", "面包", "鸡蛋"],
        ["牛奶", "面包"],
        ["面包", "黄油"],
        ["牛奶", "鸡蛋"],
        ["面包", "鸡蛋"],
        ["牛奶", "面包", "黄油"],
    ]

    model = AprioriScratch(min_support=0.3, min_confidence=0.6)
    freq = model.fit(transactions)
    rules = model.generate_rules()
    print("=== Apriori Scratch ===")
    print("频繁项集:")
    for k, v in freq.items():
        print(set(k), f"support={v:.2f}")
    print("关联规则:")
    for lhs, rhs, sup, conf in rules:
        print(f"{lhs} -> {rhs}, support={sup:.2f}, confidence={conf:.2f}")


if __name__ == "__main__":
    main()
