"""基础 smoke test：尝试运行代表性算法脚本。"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 按重构后的新目录组织代表性脚本。
SMOKE_SCRIPTS = [
    "supervised_learning/regression/linear_regression/sklearn_demo.py",
    "supervised_learning/regression/linear_regression/scratch.py",
    "supervised_learning/classification/knn/sklearn_demo.py",
    "supervised_learning/classification/knn/scratch.py",
    "unsupervised_learning/clustering/kmeans/sklearn_demo.py",
    "supervised_learning/classification/decision_tree/sklearn_demo.py",
    "supervised_learning/classification/svm/sklearn_demo.py",
    "deep_learning/mlp/sklearn_demo.py",
]


def run_script(relative_path: str, timeout_sec: int = 120) -> tuple[str, int, str]:
    """运行单个脚本并返回结果。"""
    script_path = PROJECT_ROOT / relative_path
    if not script_path.exists():
        return ("missing", 1, f"脚本不存在: {relative_path}")

    env = os.environ.copy()
    # 避免在无图形环境中卡住 matplotlib 的窗口显示。
    env["MPLBACKEND"] = "Agg"

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            env=env,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return ("timeout", 1, f"运行超时: {relative_path}")

    output = (result.stdout or "") + (result.stderr or "")
    return ("ok" if result.returncode == 0 else "fail", result.returncode, output)


def main() -> None:
    print("=== Smoke Tests: ml_algorithm_study ===")
    print(f"Python 可执行文件: {sys.executable}")
    print(f"项目根目录: {PROJECT_ROOT}")

    passed = 0
    failed = 0
    warnings = []

    for script in SMOKE_SCRIPTS:
        status, code, output = run_script(script)

        if status == "ok":
            passed += 1
            print(f"[PASS] {script}")
            # 运行成功但提示依赖缺失，也记为告警，方便人工确认。
            if any(word in output for word in ["缺少依赖", "未安装", "未检测到"]):
                warnings.append(f"{script}: 可能缺少依赖（请检查输出）")
        elif status == "missing":
            failed += 1
            print(f"[FAIL] {script} -> 文件不存在")
        elif status == "timeout":
            failed += 1
            print(f"[FAIL] {script} -> 运行超时")
        else:
            failed += 1
            print(f"[FAIL] {script} -> 退出码 {code}")
            short_out = output.strip().splitlines()
            if short_out:
                print(f"       输出: {short_out[-1]}")

    print("\n=== Smoke 测试汇总 ===")
    print(f"通过: {passed}")
    print(f"失败: {failed}")
    if warnings:
        print("告警:")
        for item in warnings:
            print(f"  - {item}")

    # 有失败时返回非零，便于 CI 或脚本判断。
    raise SystemExit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
