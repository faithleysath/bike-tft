const fs = require("fs");
const path = require("path");
const childProcess = require("child_process");
const {
  AlignmentType,
  BorderStyle,
  Document,
  Footer,
  Header,
  HeadingLevel,
  ImageRun,
  Packer,
  PageNumber,
  PageOrientation,
  Paragraph,
  SectionType,
  ShadingType,
  Table,
  TableCell,
  TableOfContents,
  TableRow,
  TextRun,
  VerticalAlign,
  WidthType,
} = require("docx");

const ROOT = path.resolve(__dirname, "..", "..");
const SOURCE_MD = path.join(ROOT, "thesis", "main_thesis_v11", "thesis_draft.md");
const FIGURE_DIR = path.join(ROOT, "thesis", "main_thesis_v11", "figures");
const TEMPLATE = path.join(ROOT, "thesis", "thesis_template.dotx");
const OUT_DIR = __dirname;
const PAGES_DIR = path.join(OUT_DIR, "pages");
const SHEETS_DIR = path.join(OUT_DIR, "contact_sheets");
const WORK_DIR = path.join(OUT_DIR, "_build");

const SHORT_TITLE = "多源特征驱动的共享单车需求预测与调度系统";
const ORIGINAL_TITLE = "多源外部特征与TFT融合的共享单车需求预测与智能调度系统";
const EN_TITLE = "A Multi-Source Feature-Driven Bike-Sharing Demand Forecasting and Rebalancing System";
const AUTHOR = "吴天一";
const AUTHOR_EN = "Wu Tianyi";
const STUDENT_ID = "202283250010";
const MAJOR = "数据科学与大数据技术";
const COLLEGE = "未来技术学院";
const ADVISOR = "胡伟";
const CN_UNIT = "南京信息工程大学未来技术学院，江苏 南京 210044";
const EN_UNIT = "School of Future Technology, NUIST, Nanjing 210044, China";
const COVER_DATE = "二○二六年五月";

const OUT_DOCX = path.join(OUT_DIR, "main_thesis_word_v11_final.docx");
const OUT_PDF = path.join(OUT_DIR, "main_thesis_word_v11_final.pdf");
const DRAFT_DOCX = path.join(WORK_DIR, "main_thesis_word_v11_pass1.docx");
const DRAFT_PDF = path.join(WORK_DIR, "main_thesis_word_v11_pass1.pdf");
const REPORT = path.join(OUT_DIR, "visual_check_report.md");
const MANIFEST = path.join(OUT_DIR, "main_thesis_word_v11_manifest.json");
const FORMULA_JSON = path.join(WORK_DIR, "formulas.json");
const POSTPROCESS_SCRIPT = path.join(OUT_DIR, "postprocess_docx_formulas.py");

const FONT_CN = { ascii: "Times New Roman", hAnsi: "Times New Roman", eastAsia: "SimSun" };
const FONT_HEI = { ascii: "Times New Roman", hAnsi: "Times New Roman", eastAsia: "SimHei" };
const FONT_KAI = { ascii: "Times New Roman", hAnsi: "Times New Roman", eastAsia: "KaiTi" };
const FONT_EN = { ascii: "Times New Roman", hAnsi: "Times New Roman", eastAsia: "Times New Roman" };
const FONT_CODE = { ascii: "Courier New", hAnsi: "Courier New", eastAsia: "SimSun" };

const PAGE = {
  portrait: {
    size: { width: 11906, height: 16838, orientation: PageOrientation.PORTRAIT },
    margin: { top: 1418, right: 1418, bottom: 1418, left: 1418, header: 851, footer: 992 },
    contentWidth: 9070,
  },
  body: {
    size: { width: 11906, height: 16838, orientation: PageOrientation.PORTRAIT },
    margin: { top: 1191, right: 1418, bottom: 1418, left: 1418, header: 851, footer: 992 },
    contentWidth: 9070,
  },
  landscape: {
    size: { width: 11906, height: 16838, orientation: PageOrientation.LANDSCAPE },
    margin: { top: 1418, right: 1418, bottom: 1418, left: 1418, header: 851, footer: 992 },
    contentWidth: 14000,
  },
};

const TABLE_CAPTIONS = [
  "表3-1 主要数据集版本",
  "表7-1 预测模型主要实验结果",
  "表7-2 外部论文模型适配对比",
  "表7-3 TFT-style 分位数预测指标",
  "表7-4 可解释性特征组敏感度",
  "表7-5 调度回测主要结果",
];

function ensureCleanDir(dir) {
  fs.rmSync(dir, { recursive: true, force: true });
  fs.mkdirSync(dir, { recursive: true });
}

function execFile(cmd, args, opts = {}) {
  return childProcess.execFileSync(cmd, args, {
    cwd: opts.cwd || ROOT,
    encoding: opts.encoding === undefined ? "utf8" : opts.encoding,
    stdio: opts.stdio || "pipe",
  });
}

function safeExecFile(cmd, args, opts = {}) {
  try {
    return execFile(cmd, args, opts);
  } catch {
    return "";
  }
}

function commandSucceeds(cmd, args, opts = {}) {
  try {
    execFile(cmd, args, opts);
    return true;
  } catch {
    return false;
  }
}

function run(text, opts = {}) {
  return new TextRun({
    text,
    bold: opts.bold,
    italics: opts.italics,
    underline: opts.underline,
    color: opts.color,
    size: opts.size || 24,
    font: opts.font || FONT_CN,
    break: opts.break,
    superScript: opts.superScript,
  });
}

function paragraph(children, opts = {}) {
  const runChildren = Array.isArray(children)
    ? children
    : [run(String(children), { size: opts.size || 24, font: opts.font || FONT_CN, bold: opts.bold })];
  return new Paragraph({
    children: runChildren,
    heading: opts.heading,
    alignment: opts.alignment,
    indent: opts.indent,
    spacing: opts.spacing || { before: 0, after: 120, line: opts.line || 360 },
    tabStops: opts.tabStops,
    border: opts.border,
    pageBreakBefore: opts.pageBreakBefore,
    keepNext: opts.keepNext,
    keepLines: opts.keepLines,
  });
}

function emptyParagraph(after = 120) {
  return paragraph("", { spacing: { before: 0, after, line: 240 } });
}

function center(text, opts = {}) {
  return paragraph(text, {
    alignment: AlignmentType.CENTER,
    heading: opts.heading,
    size: opts.size || 24,
    font: opts.font || FONT_CN,
    bold: opts.bold,
    spacing: opts.spacing || { before: 0, after: 120, line: 300 },
    pageBreakBefore: opts.pageBreakBefore,
    keepNext: opts.keepNext,
  });
}

function cleanInlineText(text) {
  return String(text)
    .replace(/\\_/g, "_")
    .replace(/\\times/g, "×")
    .replace(/\\in/g, "∈")
    .replace(/\\sum/g, "∑")
    .replace(/\\min/g, "min")
    .replace(/\\max/g, "max")
    .replace(/\\lambda/g, "λ")
    .replace(/\*\*/g, "")
    .replace(/\s+/g, " ")
    .trim();
}

function splitCitationRuns(text, opts = {}) {
  const runs = [];
  const font = opts.font || FONT_CN;
  const size = opts.size || 24;
  if (opts.disableCitationSuperscript) {
    runs.push(run(cleanInlineText(text), { size, font, bold: opts.bold }));
    return runs;
  }
  const citationPattern = /\[(\d+(?:-\d+)?(?:,\s*\d+(?:-\d+)?)*)\]/g;
  let last = 0;
  let match;
  while ((match = citationPattern.exec(text)) !== null) {
    const before = text.slice(last, match.index);
    if (before) runs.push(run(cleanInlineText(before), { size, font, bold: opts.bold }));
    runs.push(run(match[0], { size: Math.max(size - 2, 16), font: FONT_EN, superScript: true }));
    last = match.index + match[0].length;
  }
  const after = text.slice(last);
  if (after) runs.push(run(cleanInlineText(after), { size, font, bold: opts.bold }));
  return runs;
}

function inlineRuns(text, opts = {}) {
  const runs = [];
  const font = opts.font || FONT_CN;
  const size = opts.size || 24;
  const parts = String(text).split(/(`[^`]+`)/g);
  const placeholderPattern = /(@@(?:FORMULA|INLINE_MATH)_\d{3}@@)/g;
  const placeholderOnlyPattern = /^@@(?:FORMULA|INLINE_MATH)_\d{3}@@$/;
  for (const part of parts) {
    if (!part) continue;
    if (part.startsWith("`") && part.endsWith("`")) {
      runs.push(run(part.slice(1, -1), { size: Math.max(size - 2, 16), font: FONT_CODE }));
    } else {
      const mathParts = part.split(placeholderPattern);
      for (const mathPart of mathParts) {
        if (!mathPart) continue;
        if (placeholderOnlyPattern.test(mathPart)) {
          runs.push(run(mathPart, { size, font: FONT_EN }));
        } else {
          runs.push(...splitCitationRuns(mathPart, {
            size,
            font,
            bold: opts.bold,
            disableCitationSuperscript: opts.disableCitationSuperscript,
          }));
        }
      }
    }
  }
  return runs;
}

function bodyParagraph(text, opts = {}) {
  return paragraph(inlineRuns(text, { size: opts.size || 24, font: opts.font || FONT_CN }), {
    alignment: AlignmentType.JUSTIFIED,
    indent: opts.noIndent ? undefined : { firstLine: 480 },
    spacing: opts.spacing || { before: 0, after: 120, line: 360 },
  });
}

function referenceParagraph(text) {
  return paragraph(inlineRuns(text, {
    size: 24,
    font: FONT_CN,
    disableCitationSuperscript: true,
  }), {
    alignment: AlignmentType.JUSTIFIED,
    indent: { hanging: 420 },
    spacing: { before: 0, after: 80, line: 360 },
  });
}

function headingParagraph(text, level) {
  const compactText = text.replace(/\s+/g, "");
  if (compactText === "参考文献") {
    return paragraph(text, {
      heading: HeadingLevel.HEADING_1,
      size: 28,
      font: FONT_HEI,
      bold: true,
      spacing: { before: 0, after: 240, line: 360 },
      pageBreakBefore: true,
      keepNext: true,
    });
  }
  if (compactText === "致谢" || compactText === "附录") {
    return center(compactText === "致谢" ? "致   谢" : "附   录", {
      heading: HeadingLevel.HEADING_1,
      size: 28,
      font: FONT_HEI,
      bold: true,
      spacing: { before: 0, after: 360, line: 360 },
      pageBreakBefore: true,
      keepNext: true,
    });
  }
  if (level === 1) {
    return paragraph(text, {
      heading: HeadingLevel.HEADING_1,
      size: 28,
      font: FONT_HEI,
      bold: true,
      spacing: { before: 384, after: 240, line: 360 },
      keepNext: true,
    });
  }
  if (level === 2) {
    return paragraph(text, {
      heading: HeadingLevel.HEADING_2,
      size: 24,
      font: FONT_HEI,
      bold: true,
      spacing: { before: 240, after: 180, line: 360 },
      keepNext: true,
    });
  }
  return paragraph(text, {
    heading: HeadingLevel.HEADING_3,
    size: 24,
    font: FONT_CN,
    bold: false,
    spacing: { before: 180, after: 120, line: 360 },
    keepNext: true,
  });
}

function captionParagraph(text, above = false) {
  return center(text, {
    size: 21,
    font: FONT_CN,
    spacing: { before: above ? 180 : 60, after: above ? 80 : 160, line: 260 },
  });
}

function formulaParagraph(element) {
  return paragraph([
    run("\t", { size: 22, font: FONT_EN }),
    run(`@@FORMULA_${String(element.formulaIndex).padStart(3, "0")}@@`, { size: 22, font: FONT_EN }),
    run("\t", { size: 22, font: FONT_EN }),
    run(`(${element.number})`, { size: 22, font: FONT_EN }),
  ], {
    alignment: AlignmentType.LEFT,
    tabStops: [
      { type: "center", position: Math.round(PAGE.body.contentWidth / 2) },
      { type: "right", position: PAGE.body.contentWidth },
    ],
    spacing: { before: 80, after: 120, line: 300 },
  });
}

function extractLogoBuffer() {
  try {
    return execFile("unzip", ["-p", TEMPLATE, "word/media/image2.png"], { encoding: "buffer" });
  } catch {
    return null;
  }
}

function imageSize(file) {
  const out = execFile("file", [file]);
  const match = out.match(/PNG image data,\s+(\d+)\s+x\s+(\d+)/);
  if (!match) return { width: 1200, height: 800 };
  return { width: Number(match[1]), height: Number(match[2]) };
}

function imageParagraph(relativePath, altText) {
  const file = path.join(path.dirname(SOURCE_MD), relativePath);
  const { width, height } = imageSize(file);
  const targetWidth = Math.min(500, width);
  const targetHeight = Math.round((targetWidth * height) / width);
  return [
    paragraph([
      new ImageRun({
        type: "png",
        data: fs.readFileSync(file),
        transformation: { width: targetWidth, height: targetHeight },
        altText: { title: altText, description: altText, name: path.basename(relativePath) },
      }),
    ], {
      alignment: AlignmentType.CENTER,
      spacing: { before: 100, after: 40, line: 240 },
      keepNext: true,
      keepLines: true,
    }),
    captionParagraph(altText),
  ];
}

function tableCell(text, opts = {}) {
  const cellRuns = inlineRuns(String(text), {
    size: opts.size || 16,
    font: opts.font || FONT_CN,
    bold: opts.bold,
  });
  return new TableCell({
    width: { size: opts.width, type: WidthType.DXA },
    verticalAlign: VerticalAlign.CENTER,
    margins: { top: 60, bottom: 60, left: 60, right: 60 },
    borders: opts.borders,
    children: [
      new Paragraph({
        alignment: opts.alignment || AlignmentType.CENTER,
        spacing: { before: 0, after: 0, line: 220 },
        children: cellRuns,
      }),
    ],
  });
}

function thesisTable(headers, rows, caption, tableIndex) {
  const colCount = headers.length;
  const tableWidth = PAGE.portrait.contentWidth;
  const widths = estimateColumnWidths(headers, rows, tableWidth);
  const topBorder = { style: BorderStyle.SINGLE, size: 8, color: "000000" };
  const midBorder = { style: BorderStyle.SINGLE, size: 4, color: "000000" };
  const bottomBorder = { style: BorderStyle.SINGLE, size: 8, color: "000000" };
  const nil = { style: BorderStyle.NIL, size: 0, color: "FFFFFF" };
  const fontSize = colCount >= 10 ? 12 : colCount >= 7 ? 14 : 18;
  const makeBorders = (rowType) => ({
    top: rowType === "header" ? topBorder : nil,
    bottom: rowType === "header" ? midBorder : rowType === "last" ? bottomBorder : nil,
    left: nil,
    right: nil,
    insideHorizontal: nil,
    insideVertical: nil,
  });
  const allRows = [
    new TableRow({
      tableHeader: true,
      children: headers.map((h, i) =>
        tableCell(h, {
          width: widths[i],
          size: fontSize,
          bold: true,
          borders: makeBorders("header"),
          font: FONT_CN,
        }),
      ),
    }),
    ...rows.map((row, rowIndex) =>
      new TableRow({
        cantSplit: true,
        children: row.map((c, i) =>
          tableCell(c, {
            width: widths[i],
            size: fontSize,
            borders: makeBorders(rowIndex === rows.length - 1 ? "last" : "body"),
            alignment: looksNumeric(c) ? AlignmentType.CENTER : AlignmentType.LEFT,
            font: FONT_CN,
          }),
        ),
      }),
    ),
  ];
  const output = [
    captionParagraph(caption || TABLE_CAPTIONS[tableIndex] || `表${tableIndex + 1}`, true),
    new Table({
      width: { size: tableWidth, type: WidthType.DXA },
      columnWidths: widths,
      layout: "fixed",
      rows: allRows,
      borders: {
        top: nil,
        bottom: nil,
        left: nil,
        right: nil,
        insideHorizontal: nil,
        insideVertical: nil,
      },
    }),
    emptyParagraph(160),
  ];
  if (tableIndex === 0) {
    output.unshift(paragraph("", {
      pageBreakBefore: true,
      spacing: { before: 0, after: 0, line: 240 },
    }));
  }
  return output;
}

function looksNumeric(value) {
  return /^[\d,.\-+eE% ]+$/.test(String(value).trim());
}

function estimateColumnWidths(headers, rows, total) {
  const scores = headers.map((header, index) => {
    const values = [header, ...rows.map((r) => r[index] || "")];
    const longest = Math.max(...values.map((v) => stringWidth(String(v))));
    return Math.max(7, Math.min(24, longest));
  });
  const sum = scores.reduce((a, b) => a + b, 0);
  let widths = scores.map((s) => Math.max(520, Math.round((s / sum) * total)));
  let diff = total - widths.reduce((a, b) => a + b, 0);
  widths[widths.length - 1] += diff;
  return widths;
}

function stringWidth(text) {
  let width = 0;
  for (const ch of text) {
    width += /[\u4e00-\u9fff]/.test(ch) ? 2 : 1;
  }
  return width;
}

function parseMarkdownTable(lines, startIndex) {
  const tableLines = [];
  let i = startIndex;
  while (i < lines.length && /^\|.*\|$/.test(lines[i].trim())) {
    tableLines.push(lines[i].trim());
    i += 1;
  }
  const split = (line) =>
    line
      .replace(/^\|/, "")
      .replace(/\|$/, "")
      .split("|")
      .map((cell) => cell.trim());
  const headers = split(tableLines[0]);
  const rows = tableLines.slice(2).map(split);
  return { headers, rows, nextIndex: i };
}

function readSourceParts() {
  const lines = fs.readFileSync(SOURCE_MD, "utf8").split(/\r?\n/);
  const parts = {
    cnAbstract: [],
    cnKeywords: "",
    enAbstract: [],
    enKeywords: "",
    bodyLines: [],
  };
  let section = "meta";
  for (let idx = 0; idx < lines.length; idx += 1) {
    const line = lines[idx];
    if (line.startsWith("## 摘要")) {
      section = "cnAbstract";
      continue;
    }
    if (line.startsWith("## Abstract")) {
      section = "enAbstract";
      continue;
    }
    if (line.startsWith("## 目录")) {
      section = "toc";
      continue;
    }
    if (line.startsWith("## 1 ")) {
      section = "body";
    }

    if (section === "cnAbstract") {
      if (line.startsWith("关键词：")) parts.cnKeywords = line.replace(/^关键词：/, "").trim();
      else if (line.trim()) parts.cnAbstract.push(line.trim());
    } else if (section === "enAbstract") {
      if (line.startsWith("Key Words:")) parts.enKeywords = line.replace(/^Key Words:\s*/, "").trim();
      else if (line.trim()) parts.enAbstract.push(line.trim());
    } else if (section === "body") {
      parts.bodyLines.push(line);
    }
  }
  return parts;
}

function parseBodyElements(lines) {
  const elements = [];
  let paragraphBuffer = [];
  let tableIndex = 0;
  let currentChapter = 0;
  const formulaCounts = {};
  let formulaIndex = 0;

  function flushParagraph() {
    if (!paragraphBuffer.length) return;
    elements.push({ type: "paragraph", text: paragraphBuffer.join(" ") });
    paragraphBuffer = [];
  }

  for (let i = 0; i < lines.length; i += 1) {
    const raw = lines[i];
    const line = raw.trim();

    if (!line) {
      flushParagraph();
      continue;
    }

    if (line.startsWith("$$")) {
      flushParagraph();
      const formulaLines = [];
      i += 1;
      while (i < lines.length && !lines[i].trim().startsWith("$$")) {
        formulaLines.push(lines[i].trim());
        i += 1;
      }
      const chapter = currentChapter || 0;
      formulaCounts[chapter] = (formulaCounts[chapter] || 0) + 1;
      formulaIndex += 1;
      elements.push({
        type: "formula",
        text: formulaLines.join(" "),
        number: `${chapter}-${formulaCounts[chapter]}`,
        formulaIndex,
      });
      continue;
    }

    const img = line.match(/^!\[(.+?)\]\((.+?)\)$/);
    if (img) {
      flushParagraph();
      elements.push({ type: "image", alt: img[1], path: img[2] });
      continue;
    }

    const h2 = line.match(/^##\s+(.+)$/);
    if (h2) {
      flushParagraph();
      const text = h2[1].trim();
      const chapterMatch = text.match(/^(\d+)\s+/);
      if (chapterMatch) currentChapter = Number(chapterMatch[1]);
      elements.push({ type: "heading", level: 1, text });
      continue;
    }

    const h3 = line.match(/^###\s+(.+)$/);
    if (h3) {
      flushParagraph();
      const text = h3[1].trim();
      const level = /^附录[A-Z]/.test(text) ? 2 : 2;
      elements.push({ type: "heading", level, text });
      continue;
    }

    if (/^\|.*\|$/.test(line) && i + 1 < lines.length && /^\|?\s*:?-{3,}/.test(lines[i + 1].trim())) {
      flushParagraph();
      const table = parseMarkdownTable(lines, i);
      const caption = TABLE_CAPTIONS[tableIndex];
      elements.push({ type: "table", headers: table.headers, rows: table.rows, caption, tableIndex });
      tableIndex += 1;
      i = table.nextIndex - 1;
      continue;
    }

    paragraphBuffer.push(line);
  }
  flushParagraph();
  return elements;
}

function collectHeadings(elements) {
  return elements
    .filter((e) => e.type === "heading" && e.level <= 3)
    .map((e) => ({ level: e.level, text: e.text, page: undefined }));
}

function collectFormulas(elements) {
  return elements
    .filter((e) => e.type === "formula")
    .map((e) => ({
      index: e.formulaIndex,
      number: e.number,
      placeholder: `@@FORMULA_${String(e.formulaIndex).padStart(3, "0")}@@`,
      latex: formulaLatex(e.text),
    }));
}

function collectInlineFormulas(elements, startIndex = 0) {
  const formulas = [];
  let inlineIndex = 0;
  const inlineFormulaPattern = /\$([^$\n]+)\$/g;

  function replaceInlineMath(text) {
    return String(text).replace(inlineFormulaPattern, (_match, latex) => {
      inlineIndex += 1;
      const placeholder = `@@INLINE_MATH_${String(inlineIndex).padStart(3, "0")}@@`;
      formulas.push({
        index: startIndex + inlineIndex,
        placeholder,
        latex: inlineFormulaLatex(latex),
      });
      return placeholder;
    });
  }

  for (const element of elements) {
    if (element.type === "paragraph") {
      element.text = replaceInlineMath(element.text);
    } else if (element.type === "table") {
      element.headers = element.headers.map(replaceInlineMath);
      element.rows = element.rows.map((row) => row.map(replaceInlineMath));
    }
  }

  return formulas;
}

function formulaLatex(raw) {
  const text = raw.replace(/\\_/g, "_").trim();
  const replacements = new Map([
    ["L_q(y, \\hat{y}) = \\max(q(y-\\hat{y}), (q-1)(y-\\hat{y}))", "L_q(y, \\hat{y}) = \\max(q(y-\\hat{y}), (q-1)(y-\\hat{y}))"],
    ["net_flow_{t,i}=arr_count_{t,i}-dep_count_{t,i}", "\\mathrm{net\\_flow}_{t,i}=\\mathrm{arr\\_count}_{t,i}-\\mathrm{dep\\_count}_{t,i}"],
    ["X_{t-11:t} \\rightarrow Y_{t+1:t+12}", "X_{t-11:t} \\rightarrow Y_{t+1:t+12}"],
    ["X \\in R^{B \\times T_{in} \\times N \\times F}", "X \\in \\mathbb{R}^{B \\times T_{\\mathrm{in}} \\times N \\times F}"],
    ["\\hat{Y} \\in R^{B \\times T_{out} \\times N \\times 2}", "\\hat{Y} \\in \\mathbb{R}^{B \\times T_{\\mathrm{out}} \\times N \\times 2}"],
    ["\\hat{Y} \\in R^{B \\times T_{out} \\times N \\times 2 \\times 3}", "\\hat{Y} \\in \\mathbb{R}^{B \\times T_{\\mathrm{out}} \\times N \\times 2 \\times 3}"],
    ["y'=\\log(1+y)", "y'=\\log(1+y)"],
    ["Loss = MAE_{dep,arr}^{model} + \\lambda MAE_{net}^{count}", "\\mathrm{Loss}=\\mathrm{MAE}_{\\mathrm{dep,arr}}^{\\mathrm{model}}+\\lambda\\mathrm{MAE}_{\\mathrm{net}}^{\\mathrm{count}}"],
    ["net_flow_{q10}=arr_{q10}-dep_{q90}", "\\mathrm{net\\_flow}_{q10}=\\mathrm{arr}_{q10}-\\mathrm{dep}_{q90}"],
    ["net_flow_{q50}=arr_{q50}-dep_{q50}", "\\mathrm{net\\_flow}_{q50}=\\mathrm{arr}_{q50}-\\mathrm{dep}_{q50}"],
    ["net_flow_{q90}=arr_{q90}-dep_{q10}", "\\mathrm{net\\_flow}_{q90}=\\mathrm{arr}_{q90}-\\mathrm{dep}_{q10}"],
    ["lower_i=0.2 \\times capacity_i", "\\mathrm{lower}_i=0.2\\times\\mathrm{capacity}_i"],
    ["upper_i=0.8 \\times capacity_i", "\\mathrm{upper}_i=0.8\\times\\mathrm{capacity}_i"],
    ["\\hat{I}_{t+h,i}=I_{t,i}+\\sum_{k=1}^{h}\\hat{net_flow}_{t+k,i}", "\\hat{I}_{t+h,i}=I_{t,i}+\\sum_{k=1}^{h}\\widehat{\\mathrm{net\\_flow}}_{t+k,i}"],
    ["source \\rightarrow donor \\rightarrow receiver \\rightarrow sink", "\\mathrm{source}\\rightarrow\\mathrm{donor}\\rightarrow\\mathrm{receiver}\\rightarrow\\mathrm{sink}"],
    ["\\min \\sum_{i \\in D}\\sum_{j \\in R} c_{ij}x_{ij}", "\\min \\sum_{i \\in D}\\sum_{j \\in R} c_{ij}x_{ij}"],
  ]);
  return replacements.get(text) || text;
}

function inlineFormulaLatex(raw) {
  const text = String(raw).replace(/\\_/g, "_").trim();
  const replacements = new Map([
    ["\\lambda=0.10", "\\lambda=0.10"],
    ["MAE_{dep,arr}^{model}", "\\mathrm{MAE}_{\\mathrm{dep,arr}}^{\\mathrm{model}}"],
    ["MAE_{net}^{count}", "\\mathrm{MAE}_{\\mathrm{net}}^{\\mathrm{count}}"],
    ["arr-dep", "\\mathrm{arr}-\\mathrm{dep}"],
    ["x_{ij}", "x_{ij}"],
    ["c_{ij}", "c_{ij}"],
  ]);
  return replacements.get(text) || text;
}

function coverSection(logoBuffer) {
  const children = [
    paragraph([run("单位代码：", { size: 22 }), run("10300", { size: 22, underline: true })], {
      alignment: AlignmentType.LEFT,
      spacing: { before: 0, after: 200, line: 260 },
    }),
    emptyParagraph(120),
    center("本科毕业设计", { size: 36, bold: true, font: FONT_HEI, spacing: { before: 0, after: 300, line: 420 } }),
  ];
  if (logoBuffer) {
    children.push(
      paragraph([
        new ImageRun({
          type: "png",
          data: logoBuffer,
          transformation: { width: 142, height: 142 },
          altText: { title: "南京信息工程大学校徽", description: "南京信息工程大学校徽", name: "nuist_logo" },
        }),
      ], {
        alignment: AlignmentType.CENTER,
        spacing: { before: 0, after: 520, line: 240 },
      }),
    );
  } else {
    children.push(emptyParagraph(620));
  }
  children.push(
    coverField("题    目：", SHORT_TITLE, 34, true),
    emptyParagraph(260),
    coverField("学生姓名：", AUTHOR, 28, false),
    coverField("学    号：", STUDENT_ID, 28, false),
    coverField("专    业：", MAJOR, 28, false),
    coverField("学    院：", COLLEGE, 28, false),
    coverField("指导教师：", ADVISOR, 28, false),
    emptyParagraph(480),
    center(COVER_DATE, { size: 26, font: FONT_CN, spacing: { before: 0, after: 0, line: 300 } }),
  );
  return { properties: { page: PAGE.portrait }, children };
}

function coverField(label, value, size, title) {
  return paragraph([
    run(label, { size, font: FONT_CN, bold: title }),
    run(value, { size, font: title ? FONT_HEI : FONT_CN, bold: title, underline: true }),
  ], {
    alignment: AlignmentType.CENTER,
    spacing: { before: 0, after: title ? 180 : 140, line: 340 },
  });
}

function declarationSection() {
  return {
    properties: { type: SectionType.NEXT_PAGE, page: PAGE.portrait },
    children: [
      center("郑 重 声 明", { size: 28, font: FONT_HEI, bold: true, spacing: { before: 120, after: 360, line: 360 } }),
      bodyParagraph("本人以“求实 创新”的科学精神从事科学研究工作，所呈交的论文是我个人在导师指导下进行的研究工作及取得的研究成果。本论文尽我所知，所有测试、数据和相关材料均为真实有效；文中除引文和致谢内容外，未抄袭其他人或其他机构已经发表或撰写过的研究成果。与我一同工作同志对本研究所做的贡献均已在论文中作了声明并表示谢意。"),
      bodyParagraph("本人毕业论文及涉及相关资料若有不实，愿意承担一切相关的法律责任。"),
      emptyParagraph(180),
      signatureLine("论文作者签名：", "签字日期："),
      emptyParagraph(260),
      center("论文使用授权说明", { size: 28, font: FONT_HEI, bold: true, spacing: { before: 120, after: 240, line: 360 } }),
      bodyParagraph("本人授权南京信息工程大学可以保留并向国家有关部门或机构送交论文和电子文档；允许论文被查阅和借阅；可以将毕业论文的全部或部分内容编入有关数据库进行检索；可以采用影印、缩印或扫描等复制手段保存、汇编本毕业论文。不可用于任何非法用途。论文的公布（包括刊登）授权南京信息工程大学办理。"),
      emptyParagraph(180),
      signatureLine("论文作者签名：", "签字日期："),
      signatureLine("指导教师签名：", "签字日期："),
    ],
  };
}

function signatureLine(left, right) {
  return paragraph([
    run(left, { size: 24 }),
    run("________________", { size: 24 }),
    run("        "),
    run(right, { size: 24 }),
    run("________________", { size: 24 }),
  ], {
    alignment: AlignmentType.CENTER,
    spacing: { before: 0, after: 180, line: 360 },
  });
}

function romanFooter() {
  return new Footer({
    children: [
      paragraph([run("", { font: FONT_EN, size: 18 }), new TextRun({ children: [PageNumber.CURRENT], font: FONT_EN, size: 18 })], {
        alignment: AlignmentType.CENTER,
        spacing: { before: 0, after: 0, line: 240 },
      }),
    ],
  });
}

function bodyHeader() {
  return new Header({
    children: [
      paragraph("南京信息工程大学本科毕业论文（设计）", {
        alignment: AlignmentType.CENTER,
        size: 18,
        font: FONT_CN,
        spacing: { before: 0, after: 0, line: 240 },
      }),
    ],
  });
}

function bodyFooter() {
  return new Footer({
    children: [
      paragraph([new TextRun({ children: [PageNumber.CURRENT], font: FONT_EN, size: 18 })], {
        alignment: AlignmentType.CENTER,
        spacing: { before: 0, after: 0, line: 240 },
      }),
    ],
  });
}

function cnAbstractSection(parts) {
  return {
    properties: {
      type: SectionType.NEXT_PAGE,
      page: { ...PAGE.portrait, pageNumbers: { start: 1, formatType: "upperRoman" } },
    },
    footers: { default: romanFooter() },
    children: [
      center(SHORT_TITLE, { size: 32, font: FONT_HEI, bold: true, spacing: { before: 80, after: 240, line: 380 } }),
      center(AUTHOR, { size: 28, font: FONT_CN, spacing: { before: 0, after: 120, line: 320 } }),
      center(CN_UNIT, { size: 21, font: FONT_CN, spacing: { before: 0, after: 260, line: 260 } }),
      paragraph([
        run("摘要：", { size: 24, font: FONT_HEI, bold: true }),
        ...inlineRuns(parts.cnAbstract.join(""), { size: 24, font: FONT_KAI }),
      ], {
        alignment: AlignmentType.JUSTIFIED,
        spacing: { before: 0, after: 180, line: 360 },
      }),
      paragraph([
        run("关键词：", { size: 24, font: FONT_HEI, bold: true }),
        run(parts.cnKeywords, { size: 24, font: FONT_KAI }),
      ], {
        alignment: AlignmentType.LEFT,
        spacing: { before: 0, after: 0, line: 360 },
      }),
    ],
  };
}

function enAbstractSection(parts) {
  return {
    properties: {
      type: SectionType.NEXT_PAGE,
      page: { ...PAGE.portrait, pageNumbers: { formatType: "upperRoman" } },
    },
    footers: { default: romanFooter() },
    children: [
      center(EN_TITLE, { size: 32, font: FONT_EN, bold: true, spacing: { before: 80, after: 240, line: 380 } }),
      center(AUTHOR_EN, { size: 28, font: FONT_EN, spacing: { before: 0, after: 120, line: 320 } }),
      center(EN_UNIT, { size: 21, font: FONT_EN, spacing: { before: 0, after: 260, line: 260 } }),
      paragraph([
        run("Abstract: ", { size: 24, font: FONT_EN, bold: true }),
        run(parts.enAbstract.join(" "), { size: 24, font: FONT_EN }),
      ], {
        alignment: AlignmentType.JUSTIFIED,
        spacing: { before: 0, after: 180, line: 360 },
      }),
      paragraph([
        run("Key Words: ", { size: 24, font: FONT_EN, bold: true }),
        run(parts.enKeywords, { size: 24, font: FONT_EN }),
      ], {
        alignment: AlignmentType.LEFT,
        spacing: { before: 0, after: 0, line: 360 },
      }),
    ],
  };
}

function tocSection(headings, tocPageMap) {
  const cachedEntries = headings.map((h) => ({
    level: h.level,
    title: h.text,
    page: tocPageMap[h.text] || "",
  }));
  const children = [
    center("目   录", { size: 32, font: FONT_HEI, bold: true, spacing: { before: 60, after: 240, line: 380 } }),
    new TableOfContents("自动目录", {
      headingStyleRange: "1-3",
      hyperlink: true,
      hideTabAndPageNumbersInWebView: true,
      cachedEntries,
      beginDirty: true,
    }),
  ];
  return {
    properties: {
      type: SectionType.NEXT_PAGE,
      page: { ...PAGE.portrait, pageNumbers: { formatType: "upperRoman" } },
    },
    footers: { default: romanFooter() },
    children,
  };
}

function bodySection(elements) {
  const children = [];
  for (const element of elements) {
    if (element.type === "heading") {
      children.push(headingParagraph(element.text, element.level));
    } else if (element.type === "paragraph") {
      const inReferenceList = /^\[\d+\]/.test(element.text.trim());
      children.push(inReferenceList ? referenceParagraph(element.text) : bodyParagraph(element.text));
    } else if (element.type === "formula") {
      children.push(formulaParagraph(element));
    } else if (element.type === "image") {
      children.push(...imageParagraph(element.path, element.alt));
    } else if (element.type === "table") {
      children.push(...thesisTable(element.headers, element.rows, element.caption, element.tableIndex));
    }
  }
  return {
    properties: {
      type: SectionType.NEXT_PAGE,
      page: { ...PAGE.body, pageNumbers: { start: 1, formatType: "decimal" } },
    },
    headers: { default: bodyHeader() },
    footers: { default: bodyFooter() },
    children,
  };
}

function makeDocument(parts, elements, headings, tocPageMap) {
  const logo = extractLogoBuffer();
  return new Document({
    creator: "Codex",
    title: SHORT_TITLE,
    description: "南京信息工程大学本科毕业设计 Word 版",
    features: { updateFields: true },
    styles: {
      default: {
        document: {
          run: { font: FONT_CN, size: 24 },
          paragraph: { spacing: { line: 360, before: 0, after: 120 } },
        },
      },
      paragraphStyles: [
        {
          id: "Heading1",
          name: "Heading 1",
          basedOn: "Normal",
          next: "Normal",
          quickFormat: true,
          run: { size: 28, bold: true, font: FONT_HEI },
          paragraph: { spacing: { before: 384, after: 240, line: 360 }, outlineLevel: 0 },
        },
        {
          id: "Heading2",
          name: "Heading 2",
          basedOn: "Normal",
          next: "Normal",
          quickFormat: true,
          run: { size: 24, bold: true, font: FONT_HEI },
          paragraph: { spacing: { before: 240, after: 180, line: 360 }, outlineLevel: 1 },
        },
        {
          id: "Heading3",
          name: "Heading 3",
          basedOn: "Normal",
          next: "Normal",
          quickFormat: true,
          run: { size: 24, font: FONT_CN },
          paragraph: { spacing: { before: 180, after: 120, line: 360 }, outlineLevel: 2 },
        },
        {
          id: "TOC1",
          name: "TOC 1",
          basedOn: "Normal",
          next: "Normal",
          run: { size: 24, font: FONT_CN },
          paragraph: { spacing: { before: 0, after: 40, line: 280 } },
        },
        {
          id: "TOC2",
          name: "TOC 2",
          basedOn: "Normal",
          next: "Normal",
          run: { size: 24, font: FONT_CN },
          paragraph: { spacing: { before: 0, after: 40, line: 280 }, indent: { left: 420 } },
        },
        {
          id: "TOC3",
          name: "TOC 3",
          basedOn: "Normal",
          next: "Normal",
          run: { size: 24, font: FONT_CN },
          paragraph: { spacing: { before: 0, after: 40, line: 280 }, indent: { left: 840 } },
        },
      ],
    },
    sections: [
      coverSection(logo),
      declarationSection(),
      cnAbstractSection(parts),
      enAbstractSection(parts),
      tocSection(headings, tocPageMap),
      bodySection(elements),
    ],
  });
}

async function writeDocx(doc, file) {
  const buffer = await Packer.toBuffer(doc);
  fs.writeFileSync(file, buffer);
}

function postprocessFormulas(docxPath) {
  execFile("python3", [POSTPROCESS_SCRIPT, "--docx", docxPath, "--formulas", FORMULA_JSON]);
}

function convertDocxToPdf(docxPath, outPdf) {
  const outdir = path.dirname(outPdf);
  fs.mkdirSync(outdir, { recursive: true });
  const generated = path.join(outdir, `${path.basename(docxPath, ".docx")}.pdf`);
  fs.rmSync(generated, { force: true });
  execFile("libreoffice", ["--headless", "--convert-to", "pdf", "--outdir", outdir, docxPath]);
  if (generated !== outPdf) fs.renameSync(generated, outPdf);
}

function pdfPages(pdfPath) {
  const info = execFile("pdfinfo", [pdfPath]);
  const match = info.match(/^Pages:\s+(\d+)/m);
  return match ? Number(match[1]) : 0;
}

function pdfPageSize(pdfPath) {
  const info = execFile("pdfinfo", [pdfPath]);
  const match = info.match(/^Page size:\s+([0-9.]+)\s+x\s+([0-9.]+)\s+pts/m);
  return match ? { width: Number(match[1]), height: Number(match[2]) } : undefined;
}

function pdfTextPages(pdfPath) {
  const text = execFile("pdftotext", [pdfPath, "-"]);
  return text.split("\f").map((page) => page.replace(/\s+/g, " ").trim());
}

function normalizeMatchText(text) {
  return String(text).replace(/\s+/g, "").replace(/[·•]/g, "");
}

function findBodyStartPage(pages) {
  const marker = "共享单车是城市慢行交通体系的重要组成部分";
  const idx = pages.findIndex((p) => normalizeMatchText(p).includes(normalizeMatchText(marker)));
  return idx >= 0 ? idx + 1 : 1;
}

function deriveTocPages(pdfPath, headings) {
  const pages = pdfTextPages(pdfPath);
  const bodyStart = findBodyStartPage(pages);
  const map = {};
  for (const h of headings) {
    const normalizedHeading = normalizeMatchText(h.text);
    for (let i = bodyStart - 1; i < pages.length; i += 1) {
      if (normalizeMatchText(pages[i]).includes(normalizedHeading)) {
        map[h.text] = i + 1 - bodyStart + 1;
        break;
      }
    }
  }
  return { map, bodyStart, pages: pages.length };
}

function renderPages(pdfPath) {
  ensureCleanDir(PAGES_DIR);
  execFile("pdftoppm", ["-png", "-r", "140", pdfPath, path.join(PAGES_DIR, "page")]);
  const files = fs.readdirSync(PAGES_DIR).filter((f) => f.endsWith(".png")).sort();
  files.forEach((file, idx) => {
    const next = `page_${String(idx + 1).padStart(3, "0")}.png`;
    fs.renameSync(path.join(PAGES_DIR, file), path.join(PAGES_DIR, next));
  });
}

function makeContactSheets() {
  ensureCleanDir(SHEETS_DIR);
  const script = `
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
pages = sorted(Path(${JSON.stringify(PAGES_DIR)}).glob("page_*.png"))
out = Path(${JSON.stringify(SHEETS_DIR)})
thumb_w = 260
gap = 24
label_h = 34
cols = 3
rows = 4
font = ImageFont.load_default()
for sheet_idx in range((len(pages) + cols * rows - 1) // (cols * rows)):
    subset = pages[sheet_idx * cols * rows:(sheet_idx + 1) * cols * rows]
    thumbs = []
    for p in subset:
        img = Image.open(p).convert("RGB")
        scale = thumb_w / img.width
        thumb = img.resize((thumb_w, int(img.height * scale)))
        thumbs.append((p, thumb))
    cell_h = max(t.height for _, t in thumbs) + label_h
    canvas = Image.new("RGB", (cols * (thumb_w + gap) + gap, rows * (cell_h + gap) + gap), "white")
    draw = ImageDraw.Draw(canvas)
    for idx, (p, thumb) in enumerate(thumbs):
        r, c = divmod(idx, cols)
        x = gap + c * (thumb_w + gap)
        y = gap + r * (cell_h + gap)
        draw.text((x, y), p.stem.replace("_", " "), fill="black", font=font)
        canvas.paste(thumb, (x, y + label_h))
        draw.rectangle((x, y + label_h, x + thumb.width - 1, y + label_h + thumb.height - 1), outline="black")
    canvas.save(out / f"contact_sheet_{sheet_idx + 1:02d}.png")
`;
  execFile("python3", ["-c", script]);
}

function collectChecks(pdfPath, tocDerivation, headings) {
  const pageCount = pdfPages(pdfPath);
  const size = pdfPageSize(pdfPath);
  const textPages = pdfTextPages(pdfPath);
  const allText = normalizeMatchText(textPages.join("\n"));
  const missingHeadings = headings.filter((h) => !tocDerivation.map[h.text]).map((h) => h.text);
  const imageFiles = fs.readdirSync(FIGURE_DIR).filter((f) => f.endsWith(".png"));
  const docXml = execFile("unzip", ["-p", OUT_DOCX, "word/document.xml"]);
  const docxEntries = safeExecFile("unzip", ["-Z1", OUT_DOCX]).split(/\r?\n/).filter(Boolean);
  const relXml = docxEntries
    .filter((entry) => entry.endsWith(".rels"))
    .map((entry) => safeExecFile("unzip", ["-p", OUT_DOCX, entry]))
    .join("\n");
  const formulaItems = JSON.parse(fs.readFileSync(FORMULA_JSON, "utf8"));
  return {
    docxZipOk: commandSucceeds("unzip", ["-tqq", OUT_DOCX]),
    embeddedObjectCount: docxEntries.filter((entry) => entry.startsWith("word/embeddings/")).length,
    macroBinaryCount: docxEntries.filter((entry) => /vbaProject\.bin$/i.test(entry)).length,
    externalRelationshipCount: (relXml.match(/TargetMode="External"/g) || []).length,
    trackedChangeTagCount: (docXml.match(/<w:(?:ins|del|moveFrom|moveTo)\b/g) || []).length,
    commentRangeCount: (docXml.match(/<w:commentRangeStart\b/g) || []).length,
    automaticTocFieldCount: (docXml.match(/<w:instrText[^>]*>TOC\b/g) || []).length,
    updateFieldsEnabled: /<w:updateFields\b/.test(safeExecFile("unzip", ["-p", OUT_DOCX, "word/settings.xml"])),
    pageCount,
    pageSize: size,
    a4: size && Math.abs(size.width - 595.3) < 1 && Math.abs(size.height - 841.9) < 1,
    ommlFormulaCount: docXml.match(/<m:oMath\b/g)?.length || 0,
    expectedFormulaCount: formulaItems.length,
    formulaPlaceholdersRemaining: (docXml.match(/@@FORMULA_/g) || []).length,
    citationSuperscriptRuns: (docXml.match(/w:val="superscript"/g) || []).length,
    hasCoverTitle: allText.includes(normalizeMatchText(SHORT_TITLE)),
    hasDeclaration: allText.includes(normalizeMatchText("郑 重 声 明")),
    hasChineseAbstract: allText.includes(normalizeMatchText("摘要：")) && allText.includes(normalizeMatchText("关键词：")),
    hasEnglishAbstract: allText.includes(normalizeMatchText("Abstract:")) && allText.includes(normalizeMatchText("Key Words:")),
    hasReferences: allText.includes(normalizeMatchText("[33] Apache ECharts")),
    hasAcknowledgement: allText.includes(normalizeMatchText("感谢南京信息工程大学未来技术学院")),
    appendixRemoved: !allText.includes(normalizeMatchText("附录")) && !allText.includes(normalizeMatchText("附录A 主要实验结果说明")),
    bodyStartPage: tocDerivation.bodyStart,
    tocMissingHeadings: missingHeadings,
    figureSourceCount: imageFiles.length,
    renderedPageCount: fs.readdirSync(PAGES_DIR).filter((f) => f.endsWith(".png")).length,
    contactSheets: fs.readdirSync(SHEETS_DIR).filter((f) => f.endsWith(".png")).sort(),
  };
}

function writeReport(checks, tocMap) {
  const lines = [
    "# Visual Check Report",
    "",
    `Generated at: ${new Date().toISOString()}`,
    "",
    "## Automatic Checks",
    "",
    `- DOCX: \`${path.basename(OUT_DOCX)}\``,
    `- PDF: \`${path.basename(OUT_PDF)}\``,
    `- DOCX ZIP integrity: ${checks.docxZipOk ? "PASS" : "FAIL"}`,
    `- Embedded OLE objects: ${checks.embeddedObjectCount}`,
    `- Macro binaries: ${checks.macroBinaryCount}`,
    `- External relationships: ${checks.externalRelationshipCount}`,
    `- Tracked change tags: ${checks.trackedChangeTagCount}`,
    `- Comment ranges: ${checks.commentRangeCount}`,
    `- Automatic TOC field: ${checks.automaticTocFieldCount > 0 ? "PASS" : "FAIL"} (${checks.automaticTocFieldCount})`,
    `- Update fields on open: ${checks.updateFieldsEnabled ? "PASS" : "FAIL"}`,
    `- PDF pages: ${checks.pageCount}`,
    `- Page size: ${checks.pageSize ? `${checks.pageSize.width} x ${checks.pageSize.height} pts` : "unknown"}`,
    `- A4 page size: ${checks.a4 ? "PASS" : "FAIL"}`,
    `- Word OMML formulas: ${checks.ommlFormulaCount}/${checks.expectedFormulaCount}`,
    `- Formula placeholders remaining: ${checks.formulaPlaceholdersRemaining}`,
    `- Superscript citation runs: ${checks.citationSuperscriptRuns}`,
    `- Rendered page PNGs: ${checks.renderedPageCount}`,
    `- Contact sheets: ${checks.contactSheets.join(", ")}`,
    `- Body absolute start page in PDF: ${checks.bodyStartPage}`,
    `- Cover/title detected: ${checks.hasCoverTitle ? "PASS" : "FAIL"}`,
    `- Declaration detected: ${checks.hasDeclaration ? "PASS" : "FAIL"}`,
    `- Chinese abstract and keywords detected: ${checks.hasChineseAbstract ? "PASS" : "FAIL"}`,
    `- English abstract and keywords detected: ${checks.hasEnglishAbstract ? "PASS" : "FAIL"}`,
    `- References detected through [33]: ${checks.hasReferences ? "PASS" : "FAIL"}`,
    `- Acknowledgement detected: ${checks.hasAcknowledgement ? "PASS" : "FAIL"}`,
    `- Appendix removed: ${checks.appendixRemoved ? "PASS" : "FAIL"}`,
    `- TOC unresolved headings: ${checks.tocMissingHeadings.length ? checks.tocMissingHeadings.join("; ") : "none"}`,
    "",
    "## TOC Page Map",
    "",
    ...Object.entries(tocMap).map(([heading, page]) => `- ${heading}: ${page}`),
    "",
    "## Manual Visual Review Notes",
    "",
    "- Contact sheets were generated for page-by-page review.",
    "- V11 uses a Word automatic TOC field with cached page entries for PDF preview.",
    "- V11 removes the short appendix section and keeps signature lines blank because signatures are not required for this pass.",
    "- Check focus: cover, declaration/signature lines, abstracts, TOC, first body page, wide tables, figures, references, acknowledgement, and final pages.",
  ];
  fs.writeFileSync(REPORT, `${lines.join("\n")}\n`, "utf8");
}

function writeManifest(checks) {
  const manifest = {
    output_docx: path.relative(ROOT, OUT_DOCX),
    output_pdf: path.relative(ROOT, OUT_PDF),
    source_markdown: path.relative(ROOT, SOURCE_MD),
    generated_at: new Date().toISOString(),
    title_on_cover: SHORT_TITLE,
    original_task_title: ORIGINAL_TITLE,
    student: {
      name: AUTHOR,
      student_id: STUDENT_ID,
      major: MAJOR,
      college: COLLEGE,
      advisor: ADVISOR,
    },
    formatting_basis: [
      "thesis/thesis_template.dotx",
      "thesis/review_inputs/20260509_upload/附件4：毕业论文（设计）撰写规范及模板.docx",
    ],
    iteration_notes: [
      "V8 is an upload-format pass based on the v7 thesis content source.",
      "Reference, acknowledgement, and appendix sections start on separate pages.",
      "Acknowledgement and appendix headings are centered according to the school template.",
      "Signature lines are intentionally left blank for this version.",
    ],
    checks,
    artifacts: {
      page_images_dir: path.relative(ROOT, PAGES_DIR),
      contact_sheets_dir: path.relative(ROOT, SHEETS_DIR),
      visual_report: path.relative(ROOT, REPORT),
    },
  };
  fs.writeFileSync(MANIFEST, `${JSON.stringify(manifest, null, 2)}\n`, "utf8");
}

async function main() {
  fs.mkdirSync(OUT_DIR, { recursive: true });
  ensureCleanDir(WORK_DIR);

  const parts = readSourceParts();
  const elements = parseBodyElements(parts.bodyLines);
  const headings = collectHeadings(elements);
  const blockFormulas = collectFormulas(elements);
  const inlineFormulas = collectInlineFormulas(elements, blockFormulas.length);
  const formulas = [...blockFormulas, ...inlineFormulas];
  fs.writeFileSync(FORMULA_JSON, `${JSON.stringify(formulas, null, 2)}\n`, "utf8");

  const pass1 = makeDocument(parts, elements, headings, {});
  await writeDocx(pass1, DRAFT_DOCX);
  postprocessFormulas(DRAFT_DOCX);
  convertDocxToPdf(DRAFT_DOCX, DRAFT_PDF);
  const pass1Toc = deriveTocPages(DRAFT_PDF, headings);

  const finalDoc = makeDocument(parts, elements, headings, pass1Toc.map);
  await writeDocx(finalDoc, OUT_DOCX);
  postprocessFormulas(OUT_DOCX);
  convertDocxToPdf(OUT_DOCX, OUT_PDF);

  const finalToc = deriveTocPages(OUT_PDF, headings);
  const stable = JSON.stringify(pass1Toc.map) === JSON.stringify(finalToc.map);
  if (!stable) {
    const stableDoc = makeDocument(parts, elements, headings, finalToc.map);
    await writeDocx(stableDoc, OUT_DOCX);
    postprocessFormulas(OUT_DOCX);
    convertDocxToPdf(OUT_DOCX, OUT_PDF);
  }

  renderPages(OUT_PDF);
  makeContactSheets();
  const toc = deriveTocPages(OUT_PDF, headings);
  const checks = collectChecks(OUT_PDF, toc, headings);
  writeReport(checks, toc.map);
  writeManifest(checks);

  console.log(`Wrote ${OUT_DOCX}`);
  console.log(`Wrote ${OUT_PDF}`);
  console.log(`Wrote ${REPORT}`);
  console.log(`Wrote ${MANIFEST}`);
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
