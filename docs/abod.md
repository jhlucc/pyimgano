# ABOD (基于角度的异常检测) 算法
## 核心思想：正常点看邻居的角度多种多样 (方差大)，而异常点看邻居的角度则非常相似 (方差小)。

在异常检测领域，我们通常首先会想到“距离”——异常点就是那些离群体很远的点。然而，ABOD 算法独辟蹊径，它认为“角度”同样蕴含着丰富的信息。它不关心一个点离邻居有多远，而是关心从这个点出发，“看”向不同邻居时的“视野”开阔程度。这种基于几何关系的视角，使得 ABOD 在处理某些复杂数据分布时具有独特的优势。

## 情况一：正常点 (Inlier)
在一个数据点密集的区域，我们选取一个被其他点环绕的正常点 P。

<svg viewBox="0 0 300 250" class="w-full h-auto" style="background-color: #f9fafb; border: 1px solid #e5e7eb; border-radius: 0.5rem; margin: 1em 0;">
<!-- Data points cluster -->
<circle cx="150" cy="125" r="50" fill="blue" fill-opacity="0.1"/>
<circle cx="130" cy="110" r="3" fill="gray"/>
<circle cx="170" cy="140" r="3" fill="gray"/>
<circle cx="155" cy="95" r="3" fill="gray"/>
<circle cx="180" cy="115" r="3" fill="gray"/>
<circle cx="120" cy="145" r="3" fill="gray"/>
<circle cx="140" cy="150" r="3" fill="gray"/>
<circle cx="165" cy="160" r="3" fill="gray"/>
<circle cx="190" cy="130" r="3" fill="gray"/>
<circle cx="110" cy="130" r="3" fill="gray"/>
<!-- The Inlier Point P -->
<circle cx="150" cy="125" r="5" fill="blue"/>
<text x="145" y="120" font-size="12" font-weight="bold" fill="white">P</text>
<!-- Neighbors -->
<circle cx="185" cy="100" r="4" fill="green"/>
<text x="190" y="98" font-size="10">A</text>
<circle cx="115" cy="90" r="4" fill="green"/>
<text x="105" y="88" font-size="10">B</text>
<circle cx="125" cy="165" r="4" fill="orange"/>
<text x="115" y="170" font-size="10">C</text>
<circle cx="195" cy="150" r="4" fill="orange"/>
<text x="200" y="155" font-size="10">D</text>
<!-- Angle 1 (Large) -->
<line x1="150" y1="125" x2="185" y2="100" stroke="green" stroke-width="1.5" stroke-dasharray="3,3"/>
<line x1="150" y1="125" x2="115" y2="90" stroke="green" stroke-width="1.5" stroke-dasharray="3,3"/>
<path d="M 165 117 A 25 25 0 0 0 135 107" fill="none" stroke="green" stroke-width="1.5"/>
<text x="140" y="80" font-size="12" fill="green" font-weight="500">角度大</text>
<!-- Angle 2 (Also Large) -->
<line x1="150" y1="125" x2="125" y2="165" stroke="orange" stroke-width="1.5" stroke-dasharray="3,3"/>
<line x1="150" y1="125" x2="195" y2="150" stroke="orange" stroke-width="1.5" stroke-dasharray="3,3"/>
<path d="M 138 145 A 30 30 0 0 1 175 142" fill="none" stroke="orange" stroke-width="1.5"/>
<text x="140" y="190" font-size="12" fill="orange" font-weight="500">角度也很大</text>
</svg>

- 场景描述: 点 P (蓝色) 深陷在主数据群（灰色点）的中心。它的邻居们（如 A, B, C, D）从各个方向将其包围，就像一个人站在一个拥挤的广场中央，四周都是行人。

- 角度观察:

  - 从点 P 指向邻居 A 和邻居 B 的两条线，形成了一个很大的夹角。这好比站在广场中央，看向左前方的人和右前方的人，视线需要转过一个很大的角度。

  - 从点 P 指向邻居 C 和邻居 D 的两条线，也形成了一个很大的夹角。

  - 如果选取其他邻居对，可能会形成各种大小不同的角度。这些角度的差异很大，变化范围广。想象一下，您的视线可以360度地在周围的人群中穿梭，形成的夹角几乎可以是任意的。

- 数学原理: 从统计学的角度来看，如果我们收集所有由点 P 与其邻居对形成的夹角，这些角度值的分布会非常广泛，可能接近于一个均匀分布。一个分布广泛的数据集，其方差（Variance）自然就很高。

- 算法结论: 由于从点 P 出发的邻居夹角多种多样，这些角度值的方差会很大。因此，ABOD 算法判定 P 是一个正常点。这个高方差的分数是所有内部点的共同特征，形成了一个稳定的“正常”基线。

- 总结: 角度方差大 → 正常点

## 情况二：异常点 (Outlier)
现在，我们选取一个远离主数据群的异常点 O。

<svg viewBox="0 0 300 250" class="w-full h-auto" style="background-color: #f9fafb; border: 1px solid #e5e7eb; border-radius: 0.5rem; margin: 1em 0;">
<!-- Data points cluster -->
<circle cx="150" cy="125" r="50" fill="blue" fill-opacity="0.1"/>
<circle cx="130" cy="110" r="3" fill="gray"/>
<circle cx="170" cy="140" r="3" fill="gray"/>
<circle cx="155" cy="95" r="3" fill="gray"/>
<circle cx="180" cy="115" r="3" fill="gray"/>
<circle cx="120" cy="145" r="3" fill="gray"/>
<circle cx="140" cy="150" r="3" fill="gray"/>
<circle cx="165" cy="160" r="3" fill="gray"/>
<circle cx="190" cy="130" r="3" fill="gray"/>
<circle cx="110" cy="130" r="3" fill="gray"/>
<!-- The Outlier Point O -->
<circle cx="50" cy="50" r="5" fill="red"/>
<text x="45" y="45" font-size="12" font-weight="bold" fill="white">O</text>
<!-- Neighbors (all in the cluster) -->
<circle cx="115" cy="90" r="4" fill="green"/>
<text x="105" y="88" font-size="10">A</text>
<circle cx="130" cy="110" r="4" fill="green"/>
<text x="120" y="108" font-size="10">B</text>
<circle cx="110" cy="130" r="4" fill="orange"/>
<text x="100" y="135" font-size="10">C</text>
<circle cx="155" cy="95" r="4" fill="orange"/>
<text x="160" y="93" font-size="10">D</text>
<!-- Angle 1 (Small) -->
<line x1="50" y1="50" x2="115" y2="90" stroke="green" stroke-width="1.5" stroke-dasharray="3,3"/>
<line x1="50" y1="50" x2="130" y2="110" stroke="green" stroke-width="1.5" stroke-dasharray="3,3"/>
<path d="M 75 65 A 30 30 0 0 1 80 75" fill="none" stroke="green" stroke-width="1.5"/>
<text x="85" y="65" font-size="12" fill="green" font-weight="500">角度小</text>
<!-- Angle 2 (Also Small) -->
<line x1="50" y1="50" x2="110" y2="130" stroke="orange" stroke-width="1.5" stroke-dasharray="3,3"/>
<line x1="50" y1="50" x2="155" y2="95" stroke="orange" stroke-width="1.5" stroke-dasharray="3,3"/>
<path d="M 78 77 A 40 40 0 0 1 95 70" fill="none" stroke="orange" stroke-width="1.5"/>
<text x="95" y="115" font-size="12" fill="orange" font-weight="500">角度也很小</text>
</svg>

场景描述: 点 O (红色) 孤独地处在远离主数据群的位置。它的所有最近邻居（A, B, C, D）都位于远方的主数据群中。这就像一个人站在高高的山顶上，眺望山下的整个村庄。

角度观察:

从点 O 指向邻居 A 和邻居 B 的两条线，形成了一个非常小的夹角。从山顶看，村东头的房子和村西头的房子虽然相隔很远，但它们在您的视野里几乎重叠在同一个方向。

从点 O 指向邻居 C 和邻居 D 的两条线，同样也形成了一个非常小的夹角。

因为所有邻居都聚集在同一个方向，所以无论如何选取邻居对，形成的夹角都会很小且彼此相似。您的视野被极大地压缩了。

数学原理: 在这种情况下，收集到的所有夹角值都会聚集在一个非常窄的范围内，接近于零度。这样一个分布狭窄的数据集，其方差必然非常小。

算法结论: 由于从点 O 出发的邻居夹角都非常相似，这些角度值的方差会很小。这个极低的分数与正常点的高方差分数形成了鲜明对比，使得异常点能够被轻易地识别出来。

总结: 角度方差小 → 异常点

为什么这种基于角度的方法很重要？
对密度变化不敏感: 传统的基于距离或密度的方法（如DBSCAN, LOF）在处理密度不均的数据集时可能会遇到困难。而 ABOD 关注的是数据点之间的相对几何关系，而不是局部点的密集程度，因此在密度变化剧烈的场景中可能表现得更稳定。

在高维空间中的潜力: 在维度非常高的数据中，“距离”的概念会变得模糊（即“维度灾难”现象），所有点之间的距离可能都趋向于相等。然而，“角度”作为一个度量标准，在高维空间中仍然能够保持其区分度。这使得 ABOD 及其变体成为处理高维数据异常检测的一个有潜力的方向。
