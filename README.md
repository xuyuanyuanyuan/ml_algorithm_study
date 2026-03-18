# ml_algorithm_study

一个面向 Python 与机器学习初学者的算法学习项目。  
核心设计是“同一算法双版本”：
- `sklearn_demo.py`：优先跑通，快速看到效果。
- `scratch.py`：教学型简化实现，帮助理解核心原理。

---

## 1. 项目介绍

本项目按学习范式与任务类型组织，覆盖：
- 监督学习（回归/分类）
- 无监督学习（聚类/降维/关联规则）
- 强化学习
- 深度学习

每个算法目录尽量保持统一结构，方便横向对比学习。

---

## 2. 项目目录结构

```text
ml_algorithm_study
├── supervised_learning
│   ├── regression
│   │   ├── linear_regression
│   │   ├── polynomial_regression
│   │   └── lasso_regression
│   └── classification
│       ├── logistic_regression
│       ├── svm
│       ├── knn
│       ├── decision_tree
│       ├── random_forest
│       ├── naive_bayes
│       ├── xgboost
│       ├── lightgbm
│       └── catboost
├── unsupervised_learning
│   ├── clustering
│   │   ├── kmeans
│   │   ├── hierarchical_clustering
│   │   ├── dbscan
│   │   └── gmm
│   ├── dimensionality_reduction
│   │   ├── pca
│   │   ├── tsne
│   │   └── autoencoder
│   └── association_rules
│       ├── apriori
│       └── fp_growth
├── reinforcement_learning
│   ├── q_learning
│   ├── sarsa
│   ├── dqn
│   ├── ppo
│   └── a3c
├── deep_learning
│   ├── mlp
│   ├── cnn
│   ├── rnn
│   ├── lstm
│   ├── gru
│   ├── transformer
│   ├── gan
│   └── unet
├── utils
├── data
├── tests
├── README.md
├── requirements.txt
└── main.py
```

---

## 3. 四大分类说明

- 监督学习：有标签数据，目标是预测数值或类别。
- 无监督学习：无标签数据，目标是发现结构（簇、低维表示、规则）。
- 强化学习：智能体在环境中试错学习策略。
- 深度学习：以神经网络为核心，处理高维复杂模式。

---

## 4. `sklearn_demo.py` 与 `scratch.py` 的区别

- `sklearn_demo.py`：面向“快速实践”，侧重工程常用 API，帮助先建立完整流程（数据 -> 训练 -> 预测 -> 评估）。
- `scratch.py`：面向“原理理解”，侧重核心步骤拆解；复杂算法使用教学型简化版或模板版，不追求工业级完整性。

建议做法：
1. 先跑 `sklearn_demo.py` 看结果。
2. 再看 `scratch.py` 对照理解公式与步骤。
3. 修改超参数并复现实验。

---

## 5. 推荐学习顺序

1. `supervised_learning/regression/linear_regression`
2. `supervised_learning/classification/logistic_regression`
3. `supervised_learning/classification/knn`
4. `supervised_learning/classification/decision_tree` -> `random_forest`
5. `unsupervised_learning/clustering/kmeans` -> `dbscan` -> `gmm`
6. `unsupervised_learning/dimensionality_reduction/pca` -> `tsne`
7. `deep_learning/mlp` -> `cnn` -> `rnn/lstm/gru`
8. `reinforcement_learning/q_learning` -> `sarsa` -> `dqn/ppo/a3c`

---

## 6. 如何运行算法

### 6.1 安装依赖

建议 Python 版本：**3.10+**

```bash
pip install -r requirements.txt
```

### 6.2 通过项目入口查看说明

```bash
python main.py
python main.py list
python main.py smoke
```

### 6.3 运行单个算法（推荐）

示例：

```bash
python supervised_learning/regression/linear_regression/sklearn_demo.py
python supervised_learning/regression/linear_regression/scratch.py
python supervised_learning/classification/knn/sklearn_demo.py
python unsupervised_learning/clustering/kmeans/sklearn_demo.py
python deep_learning/mlp/sklearn_demo.py
```

---

## 7. 复杂算法说明（模板版 vs 完整版）

相对完整可运行（依赖安装后可直接训练）：
- `xgboost/sklearn_demo.py`
- `lightgbm/sklearn_demo.py`
- `catboost/sklearn_demo.py`
- 多数 `sklearn_demo.py`、大多数 `torch` demo

教学型模板或简化实现（重点是原理结构）：
- `xgboost/scratch.py`、`lightgbm/scratch.py`、`catboost/scratch.py`
- `tsne/scratch.py`
- `dqn/ppo/a3c` 的 `scratch.py`
- `transformer/gan/unet` 的 `scratch.py`

说明：模板版主要用于教学与代码阅读，不等价于生产级实现。

---

## 8. 基础验证脚本（Smoke Test）

已提供：
- `tests/run_smoke_tests.py`

运行方式：

```bash
python tests/run_smoke_tests.py
```

该脚本会尝试运行一组代表性算法，帮助你快速检查环境与路径是否正常。

---

## 9. 依赖与可选项说明

- 核心依赖：`numpy`、`pandas`、`matplotlib`、`scikit-learn`
- 提升树扩展：`xgboost`、`lightgbm`、`catboost`
- 深度学习：`torch`
- 强化学习：`gymnasium`、`stable-baselines3`
- 关联规则：`mlxtend`

如果你只学习传统机器学习，可先安装核心依赖；深度学习与强化学习部分可后续按需安装。
