const fs = require("fs");
const path = require("path");
const {
  AlignmentType,
  BorderStyle,
  Document,
  Footer,
  Header,
  ImageRun,
  PageNumber,
  Packer,
  Paragraph,
  ShadingType,
  Table,
  TableCell,
  TableRow,
  TextRun,
  VerticalAlign,
  WidthType,
} = require("docx");

const OUT_DIR = __dirname;
const ASSET_DIR = path.join(__dirname, "translation_draft_assets");
const OUT_DOCX = path.join(OUT_DIR, "graph_wavenet_ijcai2019_translation_draft_v1.docx");
const OUT_MANIFEST = path.join(OUT_DIR, "graph_wavenet_ijcai2019_translation_draft_v1_manifest.json");

const FONT_CN = { ascii: "Times New Roman", hAnsi: "Times New Roman", eastAsia: "SimSun" };
const FONT_HEI = { ascii: "Times New Roman", hAnsi: "Times New Roman", eastAsia: "SimHei" };
const FONT_KAI = { ascii: "Times New Roman", hAnsi: "Times New Roman", eastAsia: "KaiTi" };
const FONT_EN = { ascii: "Times New Roman", hAnsi: "Times New Roman", eastAsia: "Times New Roman" };

const page = {
  size: { width: 11906, height: 16838 },
  margin: { top: 1440, right: 1260, bottom: 1440, left: 1260 },
};

function run(text, opts = {}) {
  return new TextRun({
    text,
    font: opts.font || FONT_CN,
    size: opts.size || 24,
    bold: opts.bold,
    italics: opts.italics,
    color: opts.color,
    break: opts.break,
  });
}

function para(children, opts = {}) {
  const runs = Array.isArray(children) ? children : [run(children, opts.run || {})];
  return new Paragraph({
    children: runs,
    alignment: opts.alignment,
    indent: opts.indent,
    spacing: opts.spacing || { before: 0, after: 120, line: opts.line || 360 },
    border: opts.border,
  });
}

function blank(lines = 1) {
  return para("", { spacing: { before: 0, after: lines * 160 } });
}

function body(text) {
  return para(text, {
    indent: { firstLine: 480 },
    spacing: { before: 0, after: 120, line: 360 },
    run: { size: 24, font: FONT_CN },
  });
}

function bodyRuns(children) {
  return para(children, {
    indent: { firstLine: 480 },
    spacing: { before: 0, after: 120, line: 360 },
  });
}

function heading(text, level = 1) {
  const size = level === 1 ? 28 : level === 2 ? 26 : 24;
  return para(text, {
    spacing: { before: level === 1 ? 240 : 180, after: 120, line: 360 },
    run: { size, bold: true, font: FONT_HEI },
  });
}

function center(text, opts = {}) {
  return para(text, {
    alignment: AlignmentType.CENTER,
    spacing: opts.spacing || { before: 0, after: 120, line: 300 },
    run: { size: opts.size || 24, bold: opts.bold, font: opts.font || FONT_CN },
  });
}

function formula(text) {
  return para(text, {
    alignment: AlignmentType.CENTER,
    spacing: { before: 80, after: 120, line: 300 },
    run: { size: 22, font: FONT_EN },
  });
}

function caption(text) {
  return center(text, { size: 21, font: FONT_CN, spacing: { before: 80, after: 160, line: 260 } });
}

function image(file, width, height) {
  return new ImageRun({
    data: fs.readFileSync(path.join(ASSET_DIR, file)),
    transformation: { width, height },
    type: "png",
  });
}

function figure(file, width, height, text) {
  return [
    para([image(file, width, height)], {
      alignment: AlignmentType.CENTER,
      spacing: { before: 100, after: 40 },
    }),
    caption(text),
  ];
}

function tableCell(text, opts = {}) {
  const lines = String(text).split("\n");
  const children = [
    new Paragraph({
      alignment: opts.alignment || AlignmentType.CENTER,
      spacing: { before: 0, after: 0, line: 240 },
      children: lines.flatMap((line, idx) => [
        run(line, {
          size: opts.size || 18,
          bold: opts.bold,
          font: opts.font || FONT_CN,
        }),
        ...(idx < lines.length - 1 ? [run("", { break: 1, size: opts.size || 18 })] : []),
      ]),
    }),
  ];
  return new TableCell({
    children,
    verticalAlign: VerticalAlign.CENTER,
    width: opts.width ? { size: opts.width, type: WidthType.PERCENTAGE } : undefined,
    shading: opts.shading,
    margins: { top: 80, bottom: 80, left: 80, right: 80 },
  });
}

function simpleTable(headers, rows, opts = {}) {
  const fontSize = opts.fontSize || 18;
  const headerRow = new TableRow({
    tableHeader: true,
    children: headers.map((h) =>
      tableCell(h, {
        size: fontSize,
        bold: true,
        shading: { fill: "D9EAF7", type: ShadingType.CLEAR },
      }),
    ),
  });
  return new Table({
    width: { size: 100, type: WidthType.PERCENTAGE },
    borders: {
      top: { style: BorderStyle.SINGLE, size: 4, color: "666666" },
      bottom: { style: BorderStyle.SINGLE, size: 4, color: "666666" },
      left: { style: BorderStyle.SINGLE, size: 4, color: "666666" },
      right: { style: BorderStyle.SINGLE, size: 4, color: "666666" },
      insideHorizontal: { style: BorderStyle.SINGLE, size: 2, color: "999999" },
      insideVertical: { style: BorderStyle.SINGLE, size: 2, color: "999999" },
    },
    rows: [
      headerRow,
      ...rows.map(
        (row) =>
          new TableRow({
            children: row.map((c) => tableCell(c, { size: fontSize })),
          }),
      ),
    ],
  });
}

function bullet(text) {
  return para(text, {
    indent: { left: 420, hanging: 260 },
    spacing: { before: 0, after: 100, line: 360 },
    run: { size: 24, font: FONT_CN },
  });
}

function ref(text) {
  return para(text, {
    indent: { left: 420, hanging: 420 },
    spacing: { before: 0, after: 60, line: 300 },
    run: { size: 21, font: FONT_EN },
  });
}

const cover = [
  blank(3),
  center("本科毕业论文英文翻译", { size: 32, bold: true, font: FONT_HEI, spacing: { before: 0, after: 420 } }),
  para([image("nuist_logo.png", 118, 118)], {
    alignment: AlignmentType.CENTER,
    spacing: { before: 0, after: 520 },
  }),
  center("原文题目", { size: 30, font: FONT_HEI, spacing: { before: 0, after: 80 } }),
  center("Graph WaveNet for Deep Spatial-Temporal Graph Modeling", {
    size: 30,
    font: FONT_EN,
    spacing: { before: 0, after: 260 },
  }),
  center("译文题目", { size: 30, font: FONT_HEI, spacing: { before: 0, after: 80 } }),
  center("用于深度时空图建模的 Graph WaveNet", {
    size: 30,
    bold: true,
    font: FONT_HEI,
    spacing: { before: 0, after: 680 },
  }),
  center("学生姓名：________________", { size: 30, font: FONT_CN, spacing: { before: 0, after: 160 } }),
  center("学    号：________________", { size: 30, font: FONT_CN, spacing: { before: 0, after: 160 } }),
  center("专    业：________________", { size: 30, font: FONT_CN, spacing: { before: 0, after: 160 } }),
  center("学    院：________________", { size: 30, font: FONT_CN, spacing: { before: 0, after: 160 } }),
  center("指导教师：________________", { size: 30, font: FONT_CN, spacing: { before: 0, after: 560 } }),
  center("二○二六年四月", { size: 28, font: FONT_CN }),
];

const children = [
  center("用于深度时空图建模的 Graph WaveNet", {
    size: 32,
    bold: true,
    font: FONT_HEI,
    spacing: { before: 0, after: 180, line: 360 },
  }),
  center("Zonghan Wu¹, Shirui Pan², Guodong Long¹, Jing Jiang¹, Chengqi Zhang¹", {
    size: 24,
    bold: true,
    font: FONT_EN,
    spacing: { before: 0, after: 80, line: 300 },
  }),
  center("1 澳大利亚悉尼科技大学工程与信息技术学院人工智能中心", {
    size: 21,
    font: FONT_CN,
    spacing: { before: 0, after: 40, line: 260 },
  }),
  center("2 澳大利亚蒙纳士大学信息技术学院", {
    size: 21,
    font: FONT_CN,
    spacing: { before: 0, after: 180, line: 260 },
  }),
  bodyRuns([
    run("摘要：", { size: 24, bold: true, font: FONT_HEI }),
    run(
      "时空图建模是分析系统中各组成部分空间关系和时间趋势的重要任务。现有方法大多在固定图结构上捕获空间依赖，并假设实体之间的潜在关系是预先给定的。然而，显式图结构（关系）并不一定能够反映真实依赖；由于数据连接不完整，真实关系也可能缺失。此外，现有方法难以有效捕获时间趋势，因为这些方法所采用的 RNN 或 CNN 不能充分建模长距离时间序列。为克服这些局限，本文提出一种新的图神经网络架构 Graph WaveNet，用于时空图建模。通过设计一种新的自适应依赖矩阵，并利用节点嵌入对其进行学习，模型能够准确捕获数据中的隐藏空间依赖。借助堆叠式扩张一维卷积组件，随着层数增加，其感受野呈指数增长，因此 Graph WaveNet 能够处理很长的序列。上述两个组件被无缝整合到统一框架中，并以端到端方式学习。在 METR-LA 和 PEMS-BAY 两个公开交通网络数据集上的实验结果表明，本文算法具有优越性能。",
      { size: 24, font: FONT_KAI },
    ),
  ]),
  bodyRuns([
    run("关键词：", { size: 24, bold: true, font: FONT_HEI }),
    run("Graph WaveNet；时空图建模；图卷积网络；扩张因果卷积；交通预测", { size: 24, font: FONT_KAI }),
  ]),

  heading("1 引言"),
  body(
    "随着图神经网络的发展，时空图建模受到了越来越多关注。该任务通过假设相连节点之间存在相互依赖关系，对动态的节点级输入进行建模，如图 1 所示。时空图建模在解决复杂系统问题中具有广泛应用，例如交通速度预测 [Li et al., 2018b]、出租车需求预测 [Yao et al., 2018]、人体动作识别 [Yan et al., 2018] 以及驾驶员操作意图预判 [Jain et al., 2016]。以交通速度预测为例，城市道路上的速度传感器构成一张图，边权由两个节点之间的欧氏距离判断。由于一条道路上的交通拥堵可能导致其入口道路速度降低，因此在建模每条道路交通速度时间序列时，将交通系统的底层图结构作为节点间依赖关系的先验知识是自然的。",
  ),
  ...figure("spmodel.png", 430, 259, "图1 时空图建模。在时空图中，每个节点都有动态输入特征；目标是在给定图结构的条件下建模各节点的动态特征。"),
  body(
    "时空图建模的基本假设是，一个节点的未来信息受其历史信息以及其邻居节点历史信息的共同影响。因此，如何同时捕获空间依赖和时间依赖成为首要挑战。近期时空图建模研究主要沿着两个方向展开：一类将图卷积网络（GCN）集成到循环神经网络（RNN）中 [Seo et al., 2018; Li et al., 2018b]，另一类将其集成到卷积神经网络（CNN）中 [Yu et al., 2018; Yan et al., 2018]。这些方法已经证明，将数据图结构引入模型是有效的，但仍面临两个主要缺陷。",
  ),
  body(
    "第一，这些研究假设数据的图结构能够反映节点之间真实的依赖关系。然而，在某些情况下，两个节点之间存在连接并不意味着它们之间存在依赖；也可能存在两个节点之间确有依赖关系但图中缺少连接的情况。以推荐系统为例，在第一种情况下，两个用户虽然相连，但他们对产品可能具有截然不同的偏好；在第二种情况下，两个用户可能具有相似偏好，却并未被链接在一起。Zhang 等 [2018] 使用注意力机制调整两个相连节点之间的依赖权重，从而处理第一种情况，但没有考虑第二种情况。",
  ),
  body(
    "第二，现有时空图建模研究在学习时间依赖方面效果有限。基于 RNN 的方法在捕获长距离序列时会受到耗时的迭代传播以及梯度爆炸或消失问题的影响 [Seo et al., 2018; Li et al., 2018b; Zhang et al., 2018]。相比之下，基于 CNN 的方法具有并行计算、梯度稳定以及内存需求低等优势 [Yu et al., 2018; Yan et al., 2018]。但是，这些工作采用标准一维卷积，其感受野大小随着隐藏层数增加而线性增长，因此需要堆叠大量层才能捕获很长的序列。",
  ),
  body(
    "本文提出一种基于 CNN 的方法 Graph WaveNet，以解决上述两个缺陷。我们提出一个图卷积层，其中自适应邻接矩阵能够通过端到端监督训练从数据中学习得到。这样，自适应邻接矩阵能够保留隐藏空间依赖。受 WaveNet [Oord et al., 2016] 启发，我们采用堆叠式扩张因果卷积来捕获时间依赖。堆叠扩张因果卷积网络的感受野大小随隐藏层数增加而指数增长。在该结构支持下，Graph WaveNet 能够高效且有效地处理具有长距离时间序列的时空图数据。本文主要贡献如下：",
  ),
  bullet("• 构建了一种能够保留隐藏空间依赖的自适应邻接矩阵。所提出的自适应邻接矩阵无需任何先验知识指导，即可从数据中自动发现不可见的图结构。实验证明，当空间依赖已知存在但未被显式提供时，该方法能够改善结果。"),
  bullet("• 提出一种同时捕获时空依赖的有效且高效的框架。核心思想是将所提出的图卷积与扩张因果卷积组合起来，使每个图卷积层能够处理由不同粒度的扩张因果卷积层提取出的节点信息中的空间依赖。"),
  bullet("• 在交通数据集上评估所提出模型，并以较低计算成本取得了最先进结果。Graph WaveNet 源代码已公开发布于 https://github.com/nnzhan/Graph-WaveNet。"),

  heading("2 相关工作"),
  heading("2.1 图卷积网络", 2),
  body(
    "图卷积网络是学习图结构数据的基础模块 [Wu et al., 2019]。它们广泛应用于节点嵌入 [Pan et al., 2018]、节点分类 [Kipf and Welling, 2017]、图分类 [Ying et al., 2018]、链接预测 [Zhang and Chen, 2018] 和节点聚类 [Wang et al., 2017] 等领域。图卷积网络主要有两类，即基于谱的方法和基于空间的方法。基于谱的方法利用图谱滤波器平滑节点输入信号 [Bruna et al., 2014; Defferrard et al., 2016; Kipf and Welling, 2017]。基于空间的方法通过聚合邻域中的特征信息来提取节点的高层表示 [Atwood and Towsley, 2016; Gilmer et al., 2017; Hamilton et al., 2017]。",
  ),
  body(
    "在这些方法中，邻接矩阵被视为先验知识，并在整个训练过程中保持固定。Monti 等 [2017] 通过高斯核学习节点邻居的权重。Velickovic 等 [2017] 通过注意力机制更新节点邻居的权重。Liu 等 [2019] 提出自适应路径层，以探索节点邻域的宽度和深度。尽管这些方法假设每个邻居对中心节点的贡献不同且需要学习，但它们仍然依赖预定义图结构。Li 等 [2018a] 采用距离度量为图分类问题自适应学习图的邻接矩阵。该生成邻接矩阵以节点输入为条件；由于时空图的输入是动态的，该方法用于时空图建模时并不稳定。",
  ),
  heading("2.2 时空图网络", 2),
  body(
    "大多数时空图网络遵循两个方向，即基于 RNN 的方法和基于 CNN 的方法。较早的 RNN 方法之一使用图卷积过滤传入循环单元的输入和隐藏状态，从而捕获时空依赖 [Seo et al., 2018]。之后的工作采用扩散卷积 [Li et al., 2018b] 和注意力机制 [Zhang et al., 2018] 等不同策略提升模型性能。另有一项并行工作使用节点级 RNN 和边级 RNN 处理时间信息的不同方面 [Jain et al., 2016]。基于 RNN 的方法主要缺点是面对长序列时效率较低，并且与图卷积网络结合时梯度更容易爆炸。基于 CNN 的方法将图卷积与标准一维卷积相结合 [Yu et al., 2018; Yan et al., 2018]。尽管这些方法具有计算效率优势，但为了扩大神经网络模型的感受野，它们必须堆叠许多层或使用全局池化。",
  ),

  heading("3 方法"),
  body(
    "本节首先给出本文所研究问题的数学定义。随后描述框架的两个构建模块，即图卷积层（GCN）和时间卷积层（TCN）。二者共同作用以捕获时空依赖。最后，我们概述整体框架结构。",
  ),
  heading("3.1 问题定义", 2),
  body(
    "图表示为 G=(V,E)，其中 V 为节点集合，E 为边集合。由图得到的邻接矩阵记为 A ∈ R^{N×N}。如果 v_i、v_j ∈ V 且 (v_i,v_j) ∈ E，则 A_ij 为 1，否则为 0。在每个时间步 t，图 G 具有动态特征矩阵 X^(t) ∈ R^{N×D}。本文将特征矩阵与图信号交替使用。给定图 G 及其历史 S 步图信号，我们的问题是学习一个函数 f，用以预测未来 T 步图信号。映射关系表示为：",
  ),
  formula("[X^(t-S):t, G]  --f-->  X^(t+1):(t+T)    (1)"),
  body("其中，X^(t-S):t ∈ R^{N×D×S}，X^(t+1):(t+T) ∈ R^{N×D×T}。"),

  heading("3.2 图卷积层", 2),
  body(
    "图卷积是在给定节点结构信息的条件下提取节点特征的核心操作。Kipf 等 [2017] 提出了 Chebyshev 谱滤波器的一阶近似 [Defferrard et al., 2016]。从基于空间的角度看，该方法通过聚合和变换节点邻域信息来平滑节点信号。其优势在于：它是一个可组合层；滤波器在空间上局部化；并且支持多维输入。令带自环的归一化邻接矩阵为 Ã ∈ R^{N×N}，输入信号为 X ∈ R^{N×D}，输出为 Z ∈ R^{N×M}，模型参数矩阵为 W ∈ R^{D×M}。在 [Kipf and Welling, 2017] 中，图卷积层定义为：",
  ),
  formula("Z = Ã X W    (2)"),
  body(
    "Li 等 [2018b] 提出了一种扩散卷积层，并证明其在时空建模中有效。他们使用 K 个有限步建模图信号的扩散过程。我们将其扩散卷积层推广为式 (2) 的形式，得到：",
  ),
  formula("Z = Σ(k=0..K) P^k X W_k    (3)"),
  body(
    "其中，P^k 表示转移矩阵的幂级数。对于无向图，P = A / rowsum(A)。对于有向图，扩散过程具有前向和后向两个方向，其中前向转移矩阵 P_f = A / rowsum(A)，后向转移矩阵 P_b = A^T / rowsum(A^T)。结合前向和后向转移矩阵，扩散图卷积层可写为：",
  ),
  formula("Z = Σ(k=0..K) P_f^k X W_{k1} + P_b^k X W_{k2}    (4)"),
  bodyRuns([
    run("自适应邻接矩阵：", { size: 24, bold: true, font: FONT_HEI }),
    run(
      "本文提出自适应邻接矩阵 Ã_adp。该自适应邻接矩阵不需要任何先验知识，而是通过随机梯度下降进行端到端学习。这样，模型能够自行发现隐藏空间依赖。为实现这一点，我们随机初始化两个带可学习参数的节点嵌入字典 E_1,E_2 ∈ R^{N×c}。自适应邻接矩阵定义为：",
      { size: 24, font: FONT_CN },
    ),
  ]),
  formula("Ã_adp = SoftMax(ReLU(E_1 E_2^T))    (5)"),
  body(
    "我们将 E_1 称为源节点嵌入，将 E_2 称为目标节点嵌入。通过 E_1 与 E_2 相乘，可以得到源节点和目标节点之间的空间依赖权重。我们使用 ReLU 激活函数消除弱连接，并使用 SoftMax 函数归一化自适应邻接矩阵。因此，归一化后的自适应邻接矩阵可以被视为隐藏扩散过程的转移矩阵。结合预定义空间依赖和自学习隐藏图依赖，我们提出如下图卷积层：",
  ),
  formula("Z = Σ(k=0..K) P_f^k X W_{k1} + P_b^k X W_{k2} + Ã_apt^k X W_{k3}    (6)"),
  body("当图结构不可用时，我们建议仅使用自适应邻接矩阵来捕获隐藏空间依赖，即："),
  formula("Z = Σ(k=0..K) Ã_apt^k X W_k    (7)"),
  body(
    "需要指出的是，本文的图卷积属于基于空间的方法。虽然为保持一致性，我们将图信号与节点特征矩阵交替使用，但式 (7) 中的图卷积实际上可解释为从不同阶邻域聚合并变换特征信息。",
  ),

  heading("3.3 时间卷积层", 2),
  body(
    "我们采用扩张因果卷积 [Yu and Koltun, 2016] 作为时间卷积层（TCN），以捕获节点的时间趋势。扩张因果卷积网络通过增加层深度，能够获得指数级扩大的感受野。与基于 RNN 的方法不同，扩张因果卷积网络能够以非递归方式恰当地处理长距离序列，这有利于并行计算并缓解梯度爆炸问题。扩张因果卷积通过对输入进行零填充来保持时间因果顺序，使当前时间步的预测只依赖历史信息。作为标准一维卷积的一种特殊形式，扩张因果卷积在输入上按一定步长跳过部分值并滑动计算，如图 2 所示。",
  ),
  formula("x * f(t) = Σ(s=0..K-1) f(s) x(t - d × s)    (8)"),
  body(
    "其中，d 为扩张因子，用于控制跳跃距离。按递增顺序堆叠具有不同扩张因子的扩张因果卷积层，模型感受野会呈指数增长。这使扩张因果卷积网络能够用更少层捕获更长序列，从而节省计算资源。",
  ),
  ...figure("tcn.png", 420, 202, "图2 核大小为 2 的扩张因果卷积。当扩张因子为 k 时，它每隔 k 步选取输入，并对选中的输入应用标准一维卷积。"),
  bodyRuns([
    run("门控 TCN：", { size: 24, bold: true, font: FONT_HEI }),
    run(
      "门控机制在循环神经网络中非常关键。已有研究表明，它们同样能够有效控制时间卷积网络层间的信息流 [Dauphin et al., 2017]。简单的门控 TCN 只包含一个输出门。给定输入 X ∈ R^{N×D×S}，其形式为：",
      { size: 24, font: FONT_CN },
    ),
  ]),
  formula("h = g(Θ_1 * X + b) ⊙ σ(Θ_2 * X + c)    (9)"),
  body(
    "其中，Θ_1、Θ_2、b 和 c 为模型参数，⊙ 表示逐元素乘积，g(·) 为输出的激活函数，σ(·) 为 sigmoid 函数，用于确定传递到下一层的信息比例。我们在模型中采用门控 TCN 以学习复杂时间依赖。虽然实验中将双曲正切函数设为激活函数 g(·)，但其他形式的门控 TCN 也可以轻松嵌入本文框架，例如类似 LSTM 的门控 TCN [Kalchbrenner et al., 2016]。",
  ),

  heading("3.4 Graph WaveNet 框架", 2),
  body(
    "Graph WaveNet 框架如图 3 所示。它由堆叠的时空层和一个输出层组成。一个时空层由图卷积层（GCN）和门控时间卷积层（Gated TCN）构成，后者包含两个并行的时间卷积层（TCN-a 和 TCN-b）。通过堆叠多个时空层，Graph WaveNet 能够在不同时间层级上处理空间依赖。例如，在底层，GCN 接收短期时间信息；在顶层，GCN 处理长期时间信息。在实践中，图卷积层的输入 h 是大小为 [N,C,L] 的三维张量，其中 N 为节点数量，C 为隐藏维度，L 为序列长度。我们将图卷积层应用于每一个 h[:,:,i] ∈ R^{N×C}。",
  ),
  ...figure("graphwave1.png", 400, 479, "图3 Graph WaveNet 框架。该框架左侧由 K 个时空层组成，右侧为输出层。输入首先经线性层变换，然后传入门控时间卷积模块（Gated TCN），再进入图卷积层（GCN）。每个时空层都具有残差连接，并通过跳跃连接接入输出层。"),
  body("我们选择平均绝对误差（MAE）作为 Graph WaveNet 的训练目标，其定义为："),
  formula("L(X̂^(t+1):(t+T); Θ) = (1 / TND) Σ_i Σ_j Σ_k |X̂_jk^(t+i) - X_jk^(t+i)|    (10)"),
  body(
    "不同于 [Li et al., 2018b; Yu et al., 2018] 等以往工作，Graph WaveNet 将 X̂^(t+1):(t+T) 作为整体输出，而不是通过 T 步递归生成 X̂^(t)。这解决了训练和测试不一致的问题：模型在训练时学习一步预测，但推理时却需要产生多步预测。为此，我们人为地将 Graph WaveNet 的感受野大小设计为等于输入序列长度，使最后一个时空层输出的时间维度恰好为 1。随后，将最后一层的输出通道数设为步长 T 的因子，以得到期望的输出维度。",
  ),

  heading("4 实验"),
  body(
    "我们在 Li 等 [2018b] 发布的两个公开交通网络数据集 METR-LA 和 PEMS-BAY 上验证 Graph WaveNet。METR-LA 记录了洛杉矶县高速公路 207 个传感器四个月的交通速度统计数据。PEMS-BAY 包含旧金山湾区 325 个传感器六个月的交通速度信息。我们采用与 [Li et al., 2018b] 相同的数据预处理流程。传感器读数被聚合到 5 分钟时间窗中。节点邻接矩阵根据道路网络距离和阈值化高斯核构建 [Shuman et al., 2012]。输入使用 Z-score 归一化。数据集按时间顺序划分为 70% 训练集、10% 验证集和 20% 测试集。详细数据集统计信息见表 1。",
  ),
  simpleTable(
    ["数据集", "节点数", "边数", "时间步数"],
    [
      ["METR-LA", "207", "1515", "34272"],
      ["PEMS-BAY", "325", "2369", "52116"],
    ],
    { fontSize: 20 },
  ),
  caption("表1 METR-LA 和 PEMS-BAY 的统计摘要。"),

  heading("4.1 基线模型", 2),
  body("我们将 Graph WaveNet 与以下模型进行比较。"),
  bullet("• ARIMA：带 Kalman 滤波的自回归积分滑动平均模型 [Li et al., 2018b]。"),
  bullet("• FC-LSTM：采用全连接 LSTM 隐藏单元的循环神经网络 [Li et al., 2018b]。"),
  bullet("• WaveNet：用于序列数据的卷积网络架构 [Oord et al., 2016]。"),
  bullet("• DCRNN：扩散卷积循环神经网络 [Li et al., 2018b]，以编码器-解码器方式结合图卷积网络和循环神经网络。"),
  bullet("• GGRU：图门控循环单元网络 [Zhang et al., 2018]，属于基于循环结构的方法，并在图卷积中使用注意力机制。"),
  bullet("• STGCN：时空图卷积网络 [Yu et al., 2018]，将图卷积与一维卷积相结合。"),

  heading("4.2 实验设置", 2),
  body(
    "实验在配备 Intel(R) Core(TM) i9-7900X CPU @ 3.30GHz 和 NVIDIA Titan Xp GPU 的计算环境中进行。为覆盖输入序列长度，我们使用八层 Graph WaveNet，其扩张因子序列为 1,2,1,2,1,2,1,2。图卷积层采用式 (4)，扩散步数 K=2。节点嵌入以大小为 10 的均匀分布随机初始化。模型使用 Adam 优化器训练，初始学习率为 0.001。图卷积层输出使用 dropout，p=0.3。评估指标包括平均绝对误差（MAE）、均方根误差（RMSE）和平均绝对百分比误差（MAPE）。训练和测试均排除缺失值。",
  ),

  heading("4.3 实验结果", 2),
  body(
    "表 2 比较了 Graph WaveNet 和基线模型在 METR-LA 与 PEMS-BAY 数据集上提前 15 分钟、30 分钟和 60 分钟预测的性能。Graph WaveNet 在两个数据集上均取得了更优结果。它大幅优于 ARIMA、FC-LSTM 和 WaveNet 等时间模型。与其他时空模型相比，Graph WaveNet 明显超过先前基于卷积的方法 STGCN，同时也优于基于循环结构的方法 DCRNN 和 GGRU。相对于表 2 所示的第二优模型 GGRU，Graph WaveNet 在 15 分钟预测范围上仅有小幅提升，但在 60 分钟预测范围上提升更明显。我们认为，这是因为本文架构更擅长在每个时间阶段检测空间依赖。GGRU 使用循环架构，其中 GCN 层参数在所有循环单元之间共享。相反，Graph WaveNet 采用堆叠式时空层，其中包含具有不同参数的独立 GCN 层。因此，每个 Graph WaveNet 中的 GCN 层都能够关注自身范围内的时间输入。",
  ),
  simpleTable(
    ["数据集", "模型", "15min MAE", "15min RMSE", "15min MAPE", "30min MAE", "30min RMSE", "30min MAPE", "60min MAE", "60min RMSE", "60min MAPE"],
    [
      ["METR-LA", "ARIMA", "3.99", "8.21", "9.60%", "5.15", "10.45", "12.70%", "6.90", "13.23", "17.40%"],
      ["METR-LA", "FC-LSTM", "3.44", "6.30", "9.60%", "3.77", "7.23", "10.90%", "4.37", "8.69", "13.20%"],
      ["METR-LA", "WaveNet", "2.99", "5.89", "8.04%", "3.59", "7.28", "10.25%", "4.45", "8.93", "13.62%"],
      ["METR-LA", "DCRNN", "2.77", "5.38", "7.30%", "3.15", "6.45", "8.80%", "3.60", "7.60", "10.50%"],
      ["METR-LA", "GGRU", "2.71", "5.24", "6.99%", "3.12", "6.36", "8.56%", "3.64", "7.65", "10.62%"],
      ["METR-LA", "STGCN", "2.88", "5.74", "7.62%", "3.47", "7.24", "9.57%", "4.59", "9.40", "12.70%"],
      ["METR-LA", "Graph WaveNet", "2.69", "5.15", "6.90%", "3.07", "6.22", "8.37%", "3.53", "7.37", "10.01%"],
      ["PEMS-BAY", "ARIMA", "1.62", "3.30", "3.50%", "2.33", "4.76", "5.40%", "3.38", "6.50", "8.30%"],
      ["PEMS-BAY", "FC-LSTM", "2.05", "4.19", "4.80%", "2.20", "4.55", "5.20%", "2.37", "4.96", "5.70%"],
      ["PEMS-BAY", "WaveNet", "1.39", "3.01", "2.91%", "1.83", "4.21", "4.16%", "2.35", "5.43", "5.87%"],
      ["PEMS-BAY", "DCRNN", "1.38", "2.95", "2.90%", "1.74", "3.97", "3.90%", "2.07", "4.74", "4.90%"],
      ["PEMS-BAY", "GGRU", "-", "-", "-", "-", "-", "-", "-", "-", "-"],
      ["PEMS-BAY", "STGCN", "1.36", "2.96", "2.90%", "1.81", "4.27", "4.17%", "2.49", "5.69", "5.79%"],
      ["PEMS-BAY", "Graph WaveNet", "1.30", "2.74", "2.73%", "1.63", "3.70", "3.67%", "1.95", "4.52", "4.63%"],
    ],
    { fontSize: 14 },
  ),
  caption("表2 Graph WaveNet 与其他基线模型的性能比较。Graph WaveNet 在两个数据集上均取得最佳结果。"),
  body(
    "我们在图 4 中绘制了 Graph WaveNet 和 WaveNet 在 METR-LA 测试数据一个快照上提前 60 分钟预测值与真实值的对比。结果显示，Graph WaveNet 生成的预测比 WaveNet 更稳定。尤其是 WaveNet 产生了一个红色尖峰，明显偏离真实值。相反，Graph WaveNet 的曲线始终位于真实值中间区域。",
  ),
  ...figure("pred12.png", 430, 328, "图4 在 METR-LA 测试数据快照上，WaveNet 与 Graph WaveNet 提前 60 分钟预测曲线的比较。"),

  heading("自适应邻接矩阵的效果", 3),
  body(
    "为验证所提出自适应邻接矩阵的有效性，我们使用五种不同邻接矩阵配置对 Graph WaveNet 进行实验。表 3 给出了 12 个预测范围上的 MAE、RMSE 和 MAPE 平均得分。可以看到，仅使用自适应矩阵的模型在平均 MAE 上甚至优于仅使用前向矩阵的模型。当图结构不可用时，Graph WaveNet 仍能取得良好性能。前向-后向-自适应模型在三个评估指标上都取得最低得分。这表明，当图结构信息已给定时，加入自适应邻接矩阵可以为模型引入新的有用信息。",
  ),
  simpleTable(
    ["数据集", "模型名称", "邻接矩阵配置", "Mean MAE", "Mean RMSE", "Mean MAPE"],
    [
      ["METR-LA", "Identity", "[I]", "3.58", "7.18", "10.21%"],
      ["METR-LA", "Forward-only", "[P]", "3.13", "6.26", "8.65%"],
      ["METR-LA", "Adaptive-only", "[Ã_adp]", "3.10", "6.21", "8.68%"],
      ["METR-LA", "Forward-backward", "[P_f, P_b]", "3.08", "6.13", "8.25%"],
      ["METR-LA", "Forward-backward-adaptive", "[P_f, P_b, Ã_adp]", "3.04", "6.09", "8.23%"],
      ["PEMS-BAY", "Identity", "[I]", "1.80", "4.05", "4.18%"],
      ["PEMS-BAY", "Forward-only", "[P_f]", "1.62", "3.61", "3.72%"],
      ["PEMS-BAY", "Adaptive-only", "[Ã_adp]", "1.61", "3.63", "3.59%"],
      ["PEMS-BAY", "Forward-backward", "[P_f, P_b]", "1.59", "3.55", "3.57%"],
      ["PEMS-BAY", "Forward-backward-adaptive", "[P_f, P_b, Ã_adp]", "1.58", "3.52", "3.55%"],
    ],
    { fontSize: 16 },
  ),
  caption("表3 不同邻接矩阵配置的实验结果。前向-后向-自适应模型在两个数据集上均取得最佳结果；仅自适应模型与仅前向模型的性能几乎相同。"),
  body(
    "在图 5 中，我们进一步考察了在 METR-LA 数据集上训练的前向-后向-自适应模型所学习到的自适应邻接矩阵。根据图 5(a)，某些列比其他列具有更多高值点，例如左侧框中的第 9 列相较于右侧框中的第 47 列更明显。这表明图中的某些节点对大多数节点具有较强影响，而其他节点影响较弱。图 5(b) 证实了这一观察：节点 9 位于若干主要道路的交叉口附近，而节点 47 位于单一道路上。",
  ),
  new Table({
    width: { size: 100, type: WidthType.PERCENTAGE },
    rows: [
      new TableRow({
        children: [
          new TableCell({
            children: [
              para([image("adp3.png", 230, 179)], { alignment: AlignmentType.CENTER, spacing: { before: 0, after: 40 } }),
              caption("（a）前 50 个节点学习到的自适应邻接矩阵热力图。"),
            ],
            verticalAlign: VerticalAlign.CENTER,
          }),
          new TableCell({
            children: [
              para([image("map3.png", 210, 200)], { alignment: AlignmentType.CENTER, spacing: { before: 0, after: 40 } }),
              caption("（b）Google Maps 上标注的部分节点地理位置。"),
            ],
            verticalAlign: VerticalAlign.CENTER,
          }),
        ],
      }),
    ],
  }),
  caption("图5 学习得到的自适应邻接矩阵。"),

  heading("计算时间", 3),
  body(
    "我们在表 4 中比较了 Graph WaveNet 与 DCRNN、STGCN 在 METR-LA 数据集上的计算成本。Graph WaveNet 的训练速度约为 DCRNN 的五倍，但比 STGCN 慢约两倍。在推理阶段，我们测量每个模型在验证数据上的总耗时。Graph WaveNet 是所有模型中推理效率最高的。这是因为 Graph WaveNet 一次运行即可生成 12 个预测，而 DCRNN 和 STGCN 必须基于先前预测结果继续生成后续结果。",
  ),
  simpleTable(
    ["模型", "训练时间（秒/epoch）", "推理时间（秒）"],
    [
      ["DCRNN", "249.31", "18.73"],
      ["STGCN", "19.10", "11.37"],
      ["Graph WaveNet", "53.68", "2.27"],
    ],
    { fontSize: 20 },
  ),
  caption("表4 METR-LA 数据集上的计算成本。"),

  heading("5 结论"),
  body(
    "本文提出了一种用于时空图建模的新模型。该模型通过结合图卷积和扩张因果卷积，能够高效且有效地捕获时空依赖。我们提出了一种从数据中自动学习隐藏空间依赖的有效方法。这为时空图建模开辟了一个新方向：当系统的依赖结构未知但需要被发现时，可由模型自主学习。在两个公开交通网络数据集上，Graph WaveNet 取得了最先进结果。未来工作中，我们将研究可扩展方法，以便将 Graph WaveNet 应用于大规模数据集，并探索学习动态空间依赖的方法。",
  ),

  heading("致谢"),
  body(
    "本研究由澳大利亚政府通过澳大利亚研究理事会（ARC）资助，资助项目包括：1）LP160100630，与澳大利亚政府卫生部合作；2）LP150100671，与澳大利亚儿童与青少年研究联盟（ARACY）和澳大利亚全球商学院（GBCA）合作。",
  ),

  heading("参考文献"),
  ref("[1] Atwood, J., and Towsley, D. Diffusion-convolutional neural networks. In NIPS, pages 1993-2001, 2016."),
  ref("[2] Bruna, J., Zaremba, W., Szlam, A., and LeCun, Y. Spectral networks and locally connected networks on graphs. In ICLR, 2014."),
  ref("[3] Dauphin, Y. N., Fan, A., Auli, M., and Grangier, D. Language modeling with gated convolutional networks. In ICML, pages 933-941, 2017."),
  ref("[4] Defferrard, M., Bresson, X., and Vandergheynst, P. Convolutional neural networks on graphs with fast localized spectral filtering. In NIPS, pages 3844-3852, 2016."),
  ref("[5] Gilmer, J., Schoenholz, S. S., Riley, P. F., Vinyals, O., and Dahl, G. E. Neural message passing for quantum chemistry. In ICML, pages 1263-1272, 2017."),
  ref("[6] Hamilton, W., Ying, Z., and Leskovec, J. Inductive representation learning on large graphs. In NIPS, pages 1024-1034, 2017."),
  ref("[7] Jain, A., Zamir, A. R., Savarese, S., and Saxena, A. Structural-RNN: Deep learning on spatio-temporal graphs. In CVPR, pages 5308-5317, 2016."),
  ref("[8] Kalchbrenner, N., Espeholt, L., Simonyan, K., van den Oord, A., Graves, A., and Kavukcuoglu, K. Neural machine translation in linear time. arXiv preprint arXiv:1610.10099, 2016."),
  ref("[9] Kipf, T. N., and Welling, M. Semi-supervised classification with graph convolutional networks. In ICLR, 2017."),
  ref("[10] Li, R., Wang, S., Zhu, F., and Huang, J. Adaptive graph convolutional neural networks. In AAAI, pages 3546-3553, 2018."),
  ref("[11] Li, Y., Yu, R., Shahabi, C., and Liu, Y. Diffusion convolutional recurrent neural network: Data-driven traffic forecasting. In ICLR, 2018."),
  ref("[12] Liu, Z., Chen, C., Li, L., Zhou, J., Li, X., Song, L., and Qi, Y. Geniepath: Graph neural networks with adaptive receptive paths. In AAAI, 2019."),
  ref("[13] Monti, F., Boscaini, D., Masci, J., Rodola, E., Svoboda, J., and Bronstein, M. M. Geometric deep learning on graphs and manifolds using mixture model CNNs. In CVPR, pages 5115-5124, 2017."),
  ref("[14] Oord, A. van den, Dieleman, S., Zen, H., Simonyan, K., Vinyals, O., Graves, A., Kalchbrenner, N., Senior, A., and Kavukcuoglu, K. WaveNet: A generative model for raw audio. arXiv preprint arXiv:1609.03499, 2016."),
  ref("[15] Pan, S., Hu, R., Fung, S., Long, G., Jiang, J., and Zhang, C. Learning graph embedding with adversarial training methods. In IJCAI, 2018."),
  ref("[16] Seo, Y., Defferrard, M., Vandergheynst, P., and Bresson, X. Structured sequence modeling with graph convolutional recurrent networks. In NIPS, pages 362-373, 2018."),
  ref("[17] Shuman, D. I., Narang, S. K., Frossard, P., Ortega, A., and Vandergheynst, P. The emerging field of signal processing on graphs: Extending high-dimensional data analysis to networks and other irregular domains. arXiv preprint arXiv:1211.0053, 2012."),
  ref("[18] Velickovic, P., Cucurull, G., Casanova, A., Romero, A., Lio, P., and Bengio, Y. Graph attention networks. In ICLR, 2017."),
  ref("[19] Wang, C., Pan, S., Long, G., Zhu, X., and Jiang, J. MGAE: Marginalized graph autoencoder for graph clustering. In CIKM, pages 889-898. ACM, 2017."),
  ref("[20] Wu, Z., Pan, S., Chen, F., Long, G., Zhang, C., and Yu, P. S. A comprehensive survey on graph neural networks. arXiv preprint arXiv:1901.00596, 2019."),
  ref("[21] Yan, S., Xiong, Y., and Lin, D. Spatial temporal graph convolutional networks for skeleton-based action recognition. In AAAI, pages 3482-3489, 2018."),
  ref("[22] Yao, H., Wu, F., Ke, J., Tang, X., Jia, Y., Lu, S., Gong, P., Ye, J., and Li, Z. Deep multi-view spatial-temporal network for taxi demand prediction. In AAAI, pages 2588-2595, 2018."),
  ref("[23] Ying, Z., You, J., Morris, C., Ren, X., Hamilton, W., and Leskovec, J. Hierarchical graph representation learning with differentiable pooling. In NIPS, pages 4800-4810, 2018."),
  ref("[24] Yu, F., and Koltun, V. Multi-scale context aggregation by dilated convolutions. In ICLR, 2016."),
  ref("[25] Yu, B., Yin, H., and Zhu, Z. Spatio-temporal graph convolutional networks: A deep learning framework for traffic forecasting. In IJCAI, pages 3634-3640, 2018."),
  ref("[26] Zhang, M., and Chen, Y. Link prediction based on graph neural networks. In NIPS, pages 5165-5175, 2018."),
  ref("[27] Zhang, J., Shi, X., Xie, J., Ma, H., King, I., and Yeung, D. GAAN: Gated attention networks for learning on large and spatiotemporal graphs. arXiv preprint arXiv:1803.07294, 2018."),
];

const doc = new Document({
  creator: "Codex",
  title: "用于深度时空图建模的 Graph WaveNet",
  description: "南京信息工程大学本科毕业论文外文文献翻译初稿",
  sections: [
    {
      properties: { page },
      children: cover,
    },
    {
      properties: { page },
      headers: {
        default: new Header({
          children: [
            center("本科毕业论文英文翻译", {
              size: 18,
              font: FONT_CN,
              spacing: { before: 0, after: 0, line: 240 },
            }),
          ],
        }),
      },
      footers: {
        default: new Footer({
          children: [
            new Paragraph({
              alignment: AlignmentType.CENTER,
              children: [
                run("第 ", { size: 18, font: FONT_CN }),
                new TextRun({ children: [PageNumber.CURRENT], size: 18, font: FONT_EN }),
                run(" 页", { size: 18, font: FONT_CN }),
              ],
            }),
          ],
        }),
      },
      children,
    },
  ],
});

Packer.toBuffer(doc).then((buffer) => {
  fs.writeFileSync(OUT_DOCX, buffer);
  fs.writeFileSync(
    OUT_MANIFEST,
    JSON.stringify(
      {
        output: path.basename(OUT_DOCX),
        generated_at: new Date().toISOString(),
        source_paper: "Graph WaveNet for Deep Spatial-Temporal Graph Modeling, IJCAI 2019, DOI 10.24963/ijcai.2019/264",
        template: "../附件12：外文文献翻译撰写规范及模板.docx",
        notes: [
          "Draft v1 follows the template structure: cover, translated title, authors, affiliations, abstract, keywords, body, figures/tables, acknowledgments, references.",
          "Student name, student ID, major, college, and advisor are placeholders because the repository did not contain those fields.",
          "References are kept in English as allowed by the template.",
          "Figures were converted from the original LaTeX source PDFs into PNG files under translation_draft_assets/.",
        ],
      },
      null,
      2,
    ) + "\n",
  );
  console.log(`Wrote ${OUT_DOCX}`);
  console.log(`Wrote ${OUT_MANIFEST}`);
});
