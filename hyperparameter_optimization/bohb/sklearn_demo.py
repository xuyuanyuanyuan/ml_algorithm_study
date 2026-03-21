"""BOHB（库版依赖模板）：优先使用 HpBandSter，如依赖复杂则保留为说明模板。"""


def main():
    # 依赖说明：
    # - hpbandster：BOHB 优化器
    # - ConfigSpace：定义搜索空间
    try:
        import hpbandster.core.nameserver as hpns  # noqa: F401
        from hpbandster.optimizers import BOHB  # noqa: F401
        from ConfigSpace import ConfigurationSpace  # noqa: F401
    except ImportError:
        print("缺少依赖，请安装：pip install hpbandster ConfigSpace")
        print("说明：BOHB 通常使用 HpBandSter 实现，它把贝叶斯优化与 Hyperband 结合在一起。")
        print("当前文件保留为教学型库版模板。")
        return

    print("=== BOHB Demo Template ===")
    print("当前环境已检测到 HpBandSter 相关依赖。")
    print("由于不同版本的 BOHB 初始化方式、NameServer 配置和并行 worker 写法差异较大，")
    print("这里先保留为依赖入口模板，便于后续按本地环境补充完整实验。")
    print("建议流程：")
    print("1. 用 ConfigSpace 定义搜索空间")
    print("2. 定义 worker.compute(config, budget) 评估函数")
    print("3. 用 BOHB 在不同 budget 上做多轮筛选")


if __name__ == "__main__":
    main()
