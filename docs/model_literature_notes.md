# 站点级共享单车预测模型文献备忘

这份文档用于记录当前仓库已经筛过的一批 `arXiv` 模型文献，方便后面做三件事：

- 为阶段 3 的 AGCRN 主模型选择提供依据
- 为后续基线对比与模型级消融准备参考
- 为毕业论文的相关工作与模型选型章节留素材

## 当前任务口径

当前仓库的任务定义已经基本固定为：

- 预测粒度：站点级
- 预测目标：未来多个步长的站点流出、流入，必要时递推库存
- 静态特征：站点位置、站点容量或容量近似
- 动态特征：历史流入、流出、库存或库存近似
- 外生变量：周期时间、天气、气温

因此，文献筛选时优先看下面几类模型：

- 能处理多站点联合预测的时空模型
- 能接入静态站点属性和外生变量的模型
- 不强依赖手工固定图结构，或者至少支持自适应图
- 适合做“主模型 + 基线对比 + 模型级消融”的实验设计

## 当前最值得优先看的主模型候选

### 1. AGCRN

- 论文：`Adaptive Graph Convolutional Recurrent Network for Traffic Forecasting`
- 链接：<https://arxiv.org/abs/2007.02842>
- 时间：`2020`
- 适合本题的原因：
  - 不强依赖预定义图，适合共享单车站点这种“空间关系存在但不完全等同于道路拓扑”的场景
  - 强调节点自适应建模，比较符合不同站点行为差异明显的任务
  - 结构比更重的 Transformer 类时空模型更稳，适合本科毕设落地
  - 后续做模型级消融时，可以围绕自适应图和节点自适应参数来展开
- 当前仓库建议定位：
  - 阶段 3 主模型首选

### 2. Graph WaveNet

- 论文：`Graph WaveNet for Deep Spatial-Temporal Graph Modeling`
- 链接：<https://arxiv.org/abs/1906.00121>
- 时间：`2019`
- 适合本题的原因：
  - 是时空图预测里非常经典的强模型
  - 具备自适应邻接矩阵，能学习隐藏站点依赖
  - 膨胀时间卷积对多步预测比较友好
- 当前仓库建议定位：
  - 阶段 3 的强基线
  - 如果 AGCRN 实现推进受阻，可作为主模型备选

### 3. STGCN

- 论文：`Spatio-Temporal Graph Convolutional Networks: A Deep Learning Framework for Traffic Forecasting`
- 链接：<https://arxiv.org/abs/1709.04875>
- 时间：`2017`
- 适合本题的原因：
  - 很经典，结构清晰，便于在论文中解释
  - 训练通常比更复杂的注意力模型轻
  - 适合作为图时空模型的基础对比对象
- 当前仓库建议定位：
  - 阶段 3 图模型基线

### 4. DCRNN

- 论文：`Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting`
- 链接：<https://arxiv.org/abs/1707.01926>
- 时间：`2017`
- 适合本题的原因：
  - 是时空图预测领域非常经典的递归式基线
  - 论文影响力高，适合写相关工作时引用
  - 对“图扩散 + 时序递推”的建模思路有代表性
- 当前仓库建议定位：
  - 阶段 3 强传统深度基线

### 5. GMAN

- 论文：`GMAN: A Graph Multi-Attention Network for Traffic Prediction`
- 链接：<https://arxiv.org/abs/1911.08415>
- 时间：`2019`
- 适合本题的原因：
  - 代表图注意力类时空预测模型
  - 如果后续论文想说明“为什么没有把 Transformer 类模型作为主模型”，这篇是合适参照
- 当前仓库建议定位：
  - 进阶阅读或补充基线
  - 不作为当前第一实现优先级

## 与共享单车任务最直接相关的文献

### 1. GCNN-DDGF

- 论文：`Predicting Station-Level Bike-Sharing Demands Using Graph Convolutional Neural Network`
- 链接：<https://arxiv.org/abs/2004.08723>
- 时间：`2020`
- 相关性：
  - 直接是站点级共享单车需求预测
  - 使用 Citi Bike 数据
  - 研究重点就是“站点之间的隐藏相关性如何帮助预测”
- 对本仓库的价值：
  - 可以作为“为什么共享单车站点预测适合图模型”的直接任务支撑文献
  - 论文写作时非常适合放在相关工作部分

### 2. B-MRGNN

- 论文：`Bike Sharing Demand Prediction based on Knowledge Sharing across Modes: A Graph-based Deep Learning Approach`
- 链接：<https://arxiv.org/abs/2203.10961>
- 时间：`2022`
- 相关性：
  - 也是共享单车需求预测
  - 重点是跨交通方式信息共享，比如地铁和网约车
- 对本仓库的价值：
  - 如果后续想写“多源融合是可选扩展方向”，这篇适合引用
  - 但它依赖额外交通模态，不适合作为当前阶段 3 的主路线

### 3. 天气条件下的共享单车图模型

- 论文：`Contextual Data Integration for Bike-sharing Demand Prediction with Graph Neural Networks in Degraded Weather Conditions`
- 链接：<https://arxiv.org/abs/2412.03307>
- 时间：`2024`
- 相关性：
  - 直接讨论天气、时间嵌入等上下文信息对共享单车预测的作用
  - 证明天气类变量仍然可以作为有效外生变量
- 对本仓库的价值：
  - 适合支撑阶段 2 中天气、气温特征的合理性
  - 也能帮助论文解释“天气属于外生协变量补充，而不完全等同于特征工程主体”

## 补充模型与非图强基线

### 1. iTransformer

- 论文：`iTransformer: Inverted Transformers Are Effective for Time Series Forecasting`
- 链接：<https://arxiv.org/abs/2310.06625>
- 时间：`2023`
- 价值：
  - 适合作为“非图、多变量时间序列强基线”
  - 如果要证明图结构建模确实有价值，拿它和 AGCRN / Graph WaveNet 对比会比较有说服力

### 2. MTGNN

- 论文：`Connecting the Dots: Multivariate Time Series Forecasting with Graph Neural Networks`
- 链接：<https://arxiv.org/abs/2005.11650>
- 时间：`2020`
- 价值：
  - 适合参考“自动学习变量关系图”的思路
  - 如果后续把站点视为节点、把多维动态量视为节点特征，这篇可以作为补充阅读

### 3. STAEformer

- 论文：`STAEformer: Spatio-Temporal Adaptive Embedding Makes Vanilla Transformer SOTA for Traffic Forecasting`
- 链接：<https://arxiv.org/abs/2308.10425>
- 时间：`2023`
- 价值：
  - 代表较新的 Transformer 风格时空模型
  - 更适合作为进阶阅读，不建议作为当前毕设的第一实现优先级

## 综述文献

### GNN for Time Series Survey

- 论文：`A Survey on Graph Neural Networks for Time Series: Forecasting, Classification, Imputation, and Anomaly Detection`
- 链接：<https://arxiv.org/abs/2307.03759>
- 时间：`2023`
- 价值：
  - 非常适合写相关工作综述
  - 可以帮助把图时空预测模型分成固定图、自适应图、注意力类、Transformer 类等不同谱系

## 当前建议的论文主线

如果后面不想把战线拉太长，当前最稳的阶段 3 方案可以这样定：

- 主模型：`AGCRN`
- 图模型基线：`STGCN`、`DCRNN`
- 强基线：`Graph WaveNet`
- 非图基线：`TFT`、`iTransformer`、传统机器学习模型
- 任务相关支撑文献：`GCNN-DDGF`
- 外生变量合理性支撑：天气条件共享单车图模型那篇 `2024` 文献

## 写论文时可以怎么用

### 相关工作部分

- 先引用图时空预测经典文献：`STGCN`、`DCRNN`、`Graph WaveNet`
- 再说明自适应图的代表工作：`AGCRN`
- 接着引用共享单车任务直接相关文献：`GCNN-DDGF`
- 如果需要补上下文变量和天气合理性，再引 `2024` 天气条件共享单车图模型论文

### 模型选择部分

可以用下面这个逻辑：

1. 共享单车站点预测属于典型多站点时空预测问题
2. 站点之间存在潜在空间依赖和功能依赖
3. 共享单车站点图不像道路网络那样天然固定
4. 因此需要一个能够自动学习图结构、同时建模空间和时间依赖的模型
5. 所以选择 `AGCRN` 作为主模型

### 对比实验部分

可以按下面这个层次设计：

- 传统统计或机器学习基线
- 非图深度学习基线
- 经典图时空基线
- 主模型 `AGCRN`
- 主模型的模型级消融

## 当前注意事项

- 天气、气温仍然可以保留，但在论文表述里更适合叫“外生变量”或“上下文协变量”
- 真正的“特征工程”应更多指向在同一数据集上构造滞后、滚动统计、净流量、库存压力、容量利用率等派生特征
- 如果阶段 3 时间紧，优先保证 `AGCRN + 1~2 个图模型基线 + 1 个非图强基线`

## 后续可补

后面如果继续查文献，可以优先补下面几类：

- 共享单车库存或调度与预测联动的论文
- AGCRN 在非道路网络场景的迁移论文
- 校园或园区级共享出行预测文献
- 更适合多变量节点特征输入的时空 Transformer 文献
