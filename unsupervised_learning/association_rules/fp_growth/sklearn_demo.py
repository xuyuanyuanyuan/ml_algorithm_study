"""FP-Growth（mlxtend 版）: 更高效挖掘频繁项集。"""


def main():
    try:
        import pandas as pd
        from mlxtend.frequent_patterns import fpgrowth, association_rules
        from mlxtend.preprocessing import TransactionEncoder
    except ImportError:
        print("缺少依赖，请安装: pip install mlxtend pandas")
        return

    transactions = [
        ["啤酒", "尿布", "面包"],
        ["啤酒", "尿布", "牛奶"],
        ["尿布", "面包"],
        ["啤酒", "面包"],
        ["牛奶", "面包"],
        ["啤酒", "尿布", "面包", "牛奶"],
    ]
    te = TransactionEncoder()
    arr = te.fit(transactions).transform(transactions)
    df = pd.DataFrame(arr, columns=te.columns_)

    freq = fpgrowth(df, min_support=0.3, use_colnames=True)
    rules = association_rules(freq, metric="confidence", min_threshold=0.6)
    print("=== FP-Growth Demo ===")
    print(freq.head())
    print(rules[["antecedents", "consequents", "support", "confidence"]].head())


if __name__ == "__main__":
    main()
