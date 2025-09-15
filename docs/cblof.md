# CBLOF: 基于聚类的局部异常因子 (Clustering-Based Local Outlier Factor)

## 算法简介

CBLOF 是一种高效的、基于聚类的无监督异常检测算法。它的核心思想非常直观：**一个数据点之所以“异常”，要么是因为它落在了某个主流数据群体的边缘地带，要么是因为它自成一派，形成了一个孤立的小群体。**

该算法通过三个主要步骤来识别异常点：
1.  **数据聚类**: 首先使用一种聚类算法（如 K-Means）将数据集划分为若干个簇。
2.  **簇的分类**: 接着，根据簇的大小和分布，将这些簇智能地分为“大簇”（代表正常数据）和“小簇”（可能包含异常数据）。
3.  **分数计算**: 最后，为每个数据点计算一个异常分数，分数越高，代表其为异常点的可能性越大。

---

## 核心思想与图解

算法的精髓在于它如何根据数据点所处的“环境”来动态地评估其“异常性”。我们可以用一个简单的二维图来解释这个“为什么”。

```
                 ^ Y轴
                 |
                 |
                 |                     (小簇 S1)
                 |                        o P2
                 |                       o
                 |
    (大簇 A)     |
   * * * |
 * * C_A * * |
 * * * * * P1    |
   * * * |
                 |                             (大簇 B)
                 |                           * * * *
                 +-----------------------> X轴   * C_B *
                                             * * * *
                                               * *

图例 (Legend):
* : 属于大簇的数据点 (Data points in large clusters)
o   : 属于小簇的数据点 (Data points in small clusters)
C_A : 大簇A的中心 (Center of Large Cluster A)
C_B : 大簇B的中心 (Center of Large Cluster B)
P1  : 大簇A边缘的一个点 (A point on the fringe of Large Cluster A)
P2  : 小簇S1中的一个点 (A point in Small Cluster S1)
```

在这个图中，算法识别出了两个大簇 (A, B) 和一个小簇 (S1)。对于不同位置的点，异常分数的计算方式完全不同：

#### 情况一：主流群体的“边缘者” (点 P1)
- **环境**: `P1` 属于一个正常的**大簇 A**。
- **原因**: 它虽然属于正常群体，但位于群体的边缘，离中心 `C_A` 较远。
- **分数计算**: 其异常分数等于 **它到自己所属簇中心 `C_A` 的距离**。
  - `Score(P1) = distance(P1, C_A)`

#### 情况二：孤立的小群体成员 (点 P2)
- **环境**: `P2` 属于一个孤立的**小簇 S1**。
- **原因**: 它所在的整个群体都规模很小，并且远离任何主流数据群体。
- **分数计算**: 其异常分数等于 **它到最近的那个大簇中心的距离**。
  - `Score(P2) = min( distance(P2, C_A), distance(P2, C_B) )`

---

## 算法步骤详解

1.  **聚类 (Clustering)**
    - 使用 K-Means 或其他聚类算法将数据集 `X` 分为 `k` 个簇。
    - 每个数据点被分配一个簇标签。

2.  **划分大/小簇 (Classifying Clusters)**
    - 计算每个簇包含的数据点数量。
    - 按数量从大到小对簇进行排序。
    - 根据 `alpha` 和 `beta` 两个参数找到一个分割点，将簇划分为“大簇”和“小簇”。
      - **alpha (α)**: 定义了“大簇”至少需要包含的数据量比例。例如 `alpha=0.9` 意味着所有大簇加起来至少要包含90%的数据。
      - **beta (β)**: 定义了簇规模的“陡降”程度。例如 `beta=5` 意味着当一个簇的大小是下一个簇的5倍以上时，就认为出现了分割点。

3.  **计算异常分数 (Calculating Anomaly Scores)**
    - **对于属于小簇的点**: 计算该点到**所有大簇中心**的距离，取最小值作为其异常分数。
    - **对于属于大簇的点**: 计算该点到其**自身所属簇中心**的距离，作为其异常分数。

---

## 主要参数解析

- `n_clusters`: 指定聚类算法要形成的簇的数量 `k`。
- `contamination`: 数据集中异常点的比例估计值。这个参数用来决定划分正常/异常的阈值。
- `alpha`: 用于划分大/小簇的比例系数，取值范围 (0.5, 1)，默认 0.9。
- `beta`: 用于划分大/小簇的陡降系数，取值范围 > 1，默认 5。

---

## 总结

CBLOF 算法的优势在于它提供了一种上下文相关的异常定义。它不使用全局统一的标准，而是根据数据点所处的局部环境（是在大簇内部还是属于一个小簇）来评估其异常程度，使其能够有效地识别出不同类型的异常点。

---

## Python 代码示例

下面的代码使用了 `pyod` 库来演示 CBLOF 算法。首先，请确保您已安装了所需库：
`pip install pyod numpy matplotlib`

```python
import numpy as np
import matplotlib.pyplot as plt
from pyod.models.cblof import CBLOF

# -- 1. 生成样本数据 --
# 生成两组正常的二维高斯分布数据（代表两个大簇）
X_inliers1 = 0.3 * np.random.randn(100, 2)
X_inliers2 = 0.3 * np.random.randn(100, 2) + np.array([5, 5])
X_inliers = np.r_[X_inliers1, X_inliers2]

# 生成一组异常数据（一个小簇和一个离散点）
X_outliers = np.random.uniform(low=-4, high=4, size=(20, 2))
X_outliers[0] = np.array([-3, -3]) # 孤立小簇
X_outliers[1] = np.array([-3.1, -3.1]) # 孤立小簇
X_outliers[2] = np.array([-2.9, -3.0]) # 孤立小簇

# 组合数据
X = np.r_[X_inliers, X_outliers]

# -- 2. 初始化并拟合 CBLOF 模型 --
# 假设数据集中有大约10%的异常点
contamination_rate = 0.1

# 初始化CBLOF检测器
# n_clusters 设为3，尝试捕捉到3个主要群体
# alpha 和 beta 使用默认值
# random_state 保证结果可复现
clf = CBLOF(n_clusters=3, contamination=contamination_rate, random_state=42)
clf.fit(X)

# -- 3. 获取预测结果 --
# y_pred 是二元标签 (0: 正常, 1: 异常)
y_pred = clf.labels_
# decision_scores_ 是原始异常分数
scores = clf.decision_scores_
# 阈值
threshold = clf.threshold_

# -- 4. 可视化结果 --
plt.figure(figsize=(10, 8))
plt.title("CBLOF Anomaly Detection")

# 绘制正常点 (蓝色)
plt.scatter(X[y_pred == 0, 0], X[y_pred == 0, 1], c='blue', label='Inliers')

# 绘制异常点 (红色)
plt.scatter(X[y_pred == 1, 0], X[y_pred == 1, 1], c='red', label='Outliers')

# 绘制簇中心
cluster_centers = clf.cluster_centers_
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='black', s=100, marker='X', label='Cluster Centers')

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.grid(True)
plt.show()

# 打印一些信息
print(f"在 {len(X)} 个点中检测到 {np.sum(y_pred)} 个异常点。")
print(f"用于区分正常/异常点的阈值为: {threshold:.4f}")
```