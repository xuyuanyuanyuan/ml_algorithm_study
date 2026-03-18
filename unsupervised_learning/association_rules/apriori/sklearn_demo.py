"""Apriori（mlxtend 版）: 挖掘频繁项集与关联规则。"""


def main():
    try:
        import pandas as pd
        from mlxtend.frequent_patterns import apriori, association_rules
        from mlxtend.preprocessing import TransactionEncoder
    except ImportError:
        print("缺少依赖，请安装: pip install mlxtend pandas")
        return

    transactions = [
        ["牛奶", "面包", "鸡蛋"],
        ["牛奶", "面包"],
        ["面包", "黄油"],
        ["牛奶", "鸡蛋"],
        ["面包", "鸡蛋"],
        ["牛奶", "面包", "黄油"],
    ]
    te = TransactionEncoder()
    arr = te.fit(transactions).transform(transactions)
    df = pd.DataFrame(arr, columns=te.columns_)

    freq = apriori(df, min_support=0.3, use_colnames=True)
    rules = association_rules(freq, metric="confidence", min_threshold=0.6)
    print("=== Apriori Demo ===")
    print(freq.head())
    print(rules[["antecedents", "consequents", "support", "confidence"]].head())


if __name__ == "__main__":
    main()
