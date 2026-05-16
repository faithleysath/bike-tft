# 任务书详情

当前位置：流程管理 > 任务书详情

## 基本信息

课题名称：多源外部特征与TFT融合的共享单车需求预测与智能调度系统

选题学生：吴天一[202283250010]

指导教师：胡伟

论文性质：毕业设计

工作量：大

题目预计难易程度：较难

选题类型：软件设计型

选题来源：自拟

## 论文（设计）目标

构建多源数据驱动的共享单车需求预测模型：融合天气、POI、时间因素、站点网络效应及历史模式等外部特征，完成站点/网格级别的多步预测。

基于TFT实现可解释的多预测步长需求预测：使用 PyTorch Forecasting 的 Temporal Fusion Transformer（TFT）进行定制训练，输出 10%/50%/90% 分位数，形成可用于调度决策的“风险感知”预测区间。

实现需求预测驱动的智能再平衡调度模块：结合预测需求与实时库存，构建再平衡需求量计算逻辑，使用 OR-Tools 求解（或启发式算法生成）调度路径/任务方案，实现成本与服务水平的权衡。

实现一套可演示的端到端原型系统：包含数据管道、模型训练与推理、调度输出与可视化（热力图/路径图），支持实验复现与结果展示。

完成系统评估与论文撰写：与多个基线模型对比，进行消融实验，给出可解释性分析与调度效果评估，形成规范的毕业设计成果（论文+代码+演示）。

## 论文（设计）内容

本课题按“数据—预测—调度—系统展示—评估总结”的路线完成，主要内容如下：

### 1）数据获取与预处理（多源特征工程）

共享单车业务数据：站点基础属性（容量、区域类型/功能区等）、历史租还记录（按小时/半小时聚合）、实时库存（若无实时数据可用历史模拟或回放）。

天气特征：温度、降水、风速、天气类型（类别特征向量化/Embedding），并区分“历史实况”和“未来预报（已知未来输入）”。

POI特征：基于站点缓冲区（如500m）统计餐饮、写字楼、地铁站等数量，形成站点静态/半静态特征。

时间与事件特征：小时、星期、是否工作日、节假日、学校假期、大型活动标志等；其中节假日/日历属于已知未来输入。

网络效应特征：邻近站点库存、流入/流出量、站点间距离/邻接关系等（geopandas 计算距离矩阵或K近邻图）。

历史模式特征：过去7天同时间段需求、过去4周同天需求等季节性滞后特征。

数据清洗与对齐：缺失值处理、异常值处理（极端天气/活动日）、时空对齐（站点-时间粒度一致）、特征标准化与类别编码。

### 2）TFT需求预测模型设计与训练

任务定义：多站点多步长预测（例如预测未来1–24小时需求），以站点ID作为分组键。

特征分组（按TFT输入结构整理）：

- 静态特征：站点容量、区域类型、POI统计等；
- 时变过去特征：历史需求、历史实况天气、历史库存/流量等；
- 时变未来特征：未来时间信息（小时/星期/工作日）、节假日日历、天气预报等；

输出与损失：采用分位数回归输出 q=0.1/0.5/0.9，使用 pinball loss（分位数损失）。

训练与调参：窗口长度、预测步长、hidden size、dropout、注意力头、学习率、batch size等；采用早停、学习率调度。

可解释性：输出/可视化注意力权重或变量重要性（展示模型更关注哪些历史时段、哪些特征/站点）。

### 3）智能调度优化模块（预测驱动再平衡）

再平衡需求计算：

- 基于预测需求（中位数或风险偏好加权）与当前库存/容量，计算站点缺车/溢车程度；
- 形成调度任务集合（从溢车站取车 → 向缺车站投放）。

优化目标与约束：

- 目标：最小化调度成本（距离/时间/车辆数）、最大化需求满足（减少缺车时段/缺车站点）；
- 约束：调度车载容量、站点容量上限、时间窗（可选）、一次调度可服务站点数（可选）。

求解方案（二选一或组合）：

- OR-Tools 建模为车辆路径问题（VRP/带容量约束CVRP/带时间窗VRPTW的简化版）；
- 或 启发式算法（贪心构造 + 局部搜索改进）生成可行方案，并与OR-Tools结果对比（作为创新/工程权衡点）。

输出：调度线路、每站取/放数量、预计里程/时间、预期服务提升指标。

### 4）系统实现与可视化展示

数据管道：脚本+定时策略+可复现流程。

空间可视化：kepler.gl 绘制需求热力图、站点缺车/溢车分布、调度车辆路径与任务点。

演示形式：Streamlit网页或Notebook展示预测曲线、分位数区间、注意力可解释性、调度路径与指标对比。

### 5）实验设计与效果评估

预测评估指标：MAE、RMSE、SMAPE/MAPE（可选）、Pinball Loss、区间覆盖率（PICP）等。

调度评估指标：缺车率/缺车站点数、需求满足率、总调度距离/时间、空驶比例、单位服务提升成本等。

对比与消融：

- 至少3个基线模型（如：历史均值/季节性基线、XGBoost/LightGBM、LSTM/GRU、N-BEATS/DeepAR/TCN等选择其三）；
- 消融实验：去掉天气、去掉POI、去掉网络效应、去掉历史模式等，验证贡献。

## 指定参考文献

[1] Lim B, Arık S Ö, Loeff N, Pfister T. Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting. International Journal of Forecasting, 2021.

[2] Vaswani A, Shazeer N, Parmar N, et al. Attention Is All You Need. NeurIPS, 2017.

[3] Hochreiter S, Schmidhuber J. Long Short-Term Memory. Neural Computation, 1997.

[4] Salinas D, Flunkert V, Gasthaus J, Januschowski T. DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks. arXiv:1704.04110.

[5] Oreshkin B N, Carpov D, Chapados N, Bengio Y. N-BEATS: Neural basis expansion analysis for interpretable time series forecasting. ICLR, 2020.

[6] Taylor S J, Letham B. Forecasting at Scale（Prophet）. The American Statistician, 2018.

[7] Gama J, Žliobaitė I, Bifet A, Pechenizkiy M, Bouchachia A. A Survey on Concept Drift Adaptation. ACM Computing Surveys, 2014.

[8] McMahan H B, Moore E, Ramage D, Hampson S, y Arcas B A. Communication-Efficient Learning of Deep Networks from Decentralized Data（FedAvg）. AISTATS, 2017.

[9] Pearl J. Causality: Models, Reasoning, and Inference (2nd Edition). Cambridge University Press, 2009.

[10] Toth P, Vigo D (eds.). Vehicle Routing: Problems, Methods, and Applications (2nd Edition). SIAM, 2014.

[11] Google. OR-Tools Documentation（车辆路径问题/运筹优化建模与求解文档）.

[12] PyTorch Forecasting. PyTorch Forecasting Documentation / TFT Implementation（库文档与示例）.

[13] OpenStreetMap. OpenStreetMap Data & API（用于POI与地理数据来源说明）.

[14] Kepler.gl. Kepler.gl Documentation（空间可视化工具说明）.

## 备注

## 指导老师审核意见和审核结论

审核结论：通过

审核意见：通过

用户单位：南京信息工程大学

版权所有：南京先极科技有限公司
