"""项目入口：打印项目说明、算法分类，并支持运行示例脚本。"""

import subprocess
import sys
from pathlib import Path

# 项目根目录。
PROJECT_ROOT = Path(__file__).resolve().parent

# 用于快速演示的一组代表性脚本（相对路径）。
SMOKE_DEMO_SCRIPTS = [
    "supervised_learning/regression/linear_regression/sklearn_demo.py",
    "supervised_learning/regression/linear_regression/scratch.py",
    "supervised_learning/classification/knn/sklearn_demo.py",
    "supervised_learning/classification/knn/scratch.py",
    "unsupervised_learning/clustering/kmeans/sklearn_demo.py",
    "supervised_learning/classification/decision_tree/sklearn_demo.py",
    "supervised_learning/classification/svm/sklearn_demo.py",
    "deep_learning/mlp/sklearn_demo.py",
]


def find_algorithm_dirs():
    """扫描所有包含 sklearn_demo.py 和 scratch.py 的算法目录。"""
    algo_dirs = []
    for base in [
        PROJECT_ROOT / "supervised_learning",
        PROJECT_ROOT / "unsupervised_learning",
        PROJECT_ROOT / "reinforcement_learning",
        PROJECT_ROOT / "deep_learning",
    ]:
        if not base.exists():
            continue
        for folder in base.rglob("*"):
            if not folder.is_dir():
                continue
            sk = folder / "sklearn_demo.py"
            sc = folder / "scratch.py"
            if sk.exists() and sc.exists():
                algo_dirs.append(folder.relative_to(PROJECT_ROOT).as_posix())
    return sorted(algo_dirs)


def print_project_intro():
    """打印项目说明与学习建议。"""
    print("=== ml_algorithm_study 项目说明 ===")
    print("这是一个面向初学者的机器学习算法学习项目。")
    print("每个算法目录通常包含两个文件：")
    print("  - sklearn_demo.py : 先快速跑通，理解“怎么用”")
    print("  - scratch.py      : 再看简化实现，理解“为什么这样做”")
    print("")
    print("建议学习顺序：")
    print("  1) supervised_learning")
    print("  2) unsupervised_learning")
    print("  3) deep_learning")
    print("  4) reinforcement_learning")


def print_categories():
    """打印当前算法分类。"""
    algo_dirs = find_algorithm_dirs()
    print("\n=== 当前可学习算法目录 ===")
    for path in algo_dirs:
        print(f"  - {path}")


def print_usage():
    """打印命令行使用方式。"""
    print("\n=== 运行方式 ===")
    print("  python main.py                       # 显示项目说明和算法列表")
    print("  python main.py list                  # 仅列出算法目录")
    print("  python main.py smoke                 # 运行代表性脚本")
    print("  python main.py <脚本相对路径>          # 运行指定脚本")
    print("")
    print("示例：")
    print("  python main.py supervised_learning/regression/linear_regression/sklearn_demo.py")
    print("  python main.py deep_learning/mlp/scratch.py")


def run_one_script(relative_path):
    """运行单个脚本。"""
    script_path = PROJECT_ROOT / relative_path
    if not script_path.exists():
        print(f"脚本不存在: {relative_path}")
        return 1

    print(f"\n>>> 正在运行: {relative_path}")
    result = subprocess.run([sys.executable, str(script_path)], check=False)
    return result.returncode


def run_smoke_set():
    """运行一组代表性 smoke 脚本。"""
    fail_count = 0
    for script in SMOKE_DEMO_SCRIPTS:
        code = run_one_script(script)
        if code != 0:
            fail_count += 1
    print(f"\nsmoke 运行完成，失败数量: {fail_count}")
    print("如需更完整检查，请运行 tests/run_smoke_tests.py")


def main():
    if len(sys.argv) == 1:
        print_project_intro()
        print_categories()
        print_usage()
        return

    arg = sys.argv[1].strip().lower()
    if arg == "list":
        print_categories()
        return
    if arg == "smoke":
        run_smoke_set()
        return

    run_one_script(sys.argv[1])


if __name__ == "__main__":
    main()
