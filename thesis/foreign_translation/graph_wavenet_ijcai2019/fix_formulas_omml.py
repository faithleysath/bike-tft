#!/usr/bin/env python3
"""Replace linear formula paragraphs in the translation draft with Word OMML.

The v1 draft intentionally kept formulas as plain centered text so the first
document could be generated quickly. This post-process step converts LaTeX
math through Pandoc, extracts the generated OMML, and writes a v2 DOCX without
mutating v1.
"""

from __future__ import annotations

import html
import json
import re
import shutil
import subprocess
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "graph_wavenet_ijcai2019_translation_draft_v1.docx"
OUTPUT = ROOT / "graph_wavenet_ijcai2019_translation_draft_v2.docx"
MANIFEST = ROOT / "graph_wavenet_ijcai2019_translation_draft_v2_manifest.json"

FORMULAS = [
    (
        "[X^(t-S):t, G]  --f-->  X^(t+1):(t+T)    (1)",
        r"[\mathbf{X}^{(t-S):t},G] \xrightarrow{f} \mathbf{X}^{(t+1):(t+T)}\, (1)",
    ),
    (
        "Z = Ã X W    (2)",
        r"\mathbf{Z}=\tilde{\mathbf{A}}\mathbf{X}\mathbf{W}\, (2)",
    ),
    (
        "Z = Σ(k=0..K) P^k X W_k    (3)",
        r"\mathbf{Z}=\sum_{k=0}^{K}\mathbf{P}^{k}\mathbf{X}\mathbf{W}_{k}\, (3)",
    ),
    (
        "Z = Σ(k=0..K) P_f^k X W_{k1} + P_b^k X W_{k2}    (4)",
        r"\mathbf{Z}=\sum_{k=0}^{K}\mathbf{P}_{f}^{k}\mathbf{X}\mathbf{W}_{k1}+\mathbf{P}_{b}^{k}\mathbf{X}\mathbf{W}_{k2}\, (4)",
    ),
    (
        "Ã_adp = SoftMax(ReLU(E_1 E_2^T))    (5)",
        r"\tilde{\mathbf{A}}_{\mathrm{adp}}=\operatorname{SoftMax}(\operatorname{ReLU}(\mathbf{E}_{1}\mathbf{E}_{2}^{T}))\, (5)",
    ),
    (
        "Z = Σ(k=0..K) P_f^k X W_{k1} + P_b^k X W_{k2} + Ã_apt^k X W_{k3}    (6)",
        r"\mathbf{Z}=\sum_{k=0}^{K}\mathbf{P}_{f}^{k}\mathbf{X}\mathbf{W}_{k1}+\mathbf{P}_{b}^{k}\mathbf{X}\mathbf{W}_{k2}+\tilde{\mathbf{A}}_{\mathrm{apt}}^{k}\mathbf{X}\mathbf{W}_{k3}\, (6)",
    ),
    (
        "Z = Σ(k=0..K) Ã_apt^k X W_k    (7)",
        r"\mathbf{Z}=\sum_{k=0}^{K}\tilde{\mathbf{A}}_{\mathrm{apt}}^{k}\mathbf{X}\mathbf{W}_{k}\, (7)",
    ),
    (
        "x * f(t) = Σ(s=0..K-1) f(s) x(t - d × s)    (8)",
        r"\mathbf{x}\star\mathbf{f}(t)=\sum_{s=0}^{K-1}\mathbf{f}(s)\mathbf{x}(t-d\times s)\, (8)",
    ),
    (
        "h = g(Θ_1 * X + b) ⊙ σ(Θ_2 * X + c)    (9)",
        r"\mathbf{h}=g(\mathbf{\Theta}_{1}\star\mathbf{\mathcal{X}}+\mathbf{b})\odot\sigma(\mathbf{\Theta}_{2}\star\mathbf{\mathcal{X}}+\mathbf{c})\, (9)",
    ),
    (
        "L(X̂^(t+1):(t+T); Θ) = (1 / TND) Σ_i Σ_j Σ_k |X̂_jk^(t+i) - X_jk^(t+i)|    (10)",
        r"L(\hat{\mathbf{X}}^{(t+1):(t+T)};\mathbf{\Theta})=\frac{1}{TND}\sum_{i=1}^{T}\sum_{j=1}^{N}\sum_{k=1}^{D}|\hat{\mathbf{X}}_{jk}^{(t+i)}-\mathbf{X}_{jk}^{(t+i)}|\, (10)",
    ),
]


def make_omml_paragraphs() -> list[str]:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        markdown = "\n\n".join(f"$$\n{latex}\n$$" for _, latex in FORMULAS)
        md_path = tmp / "formulas.md"
        docx_path = tmp / "formulas.docx"
        md_path.write_text(markdown, encoding="utf-8")
        subprocess.run(["pandoc", str(md_path), "-o", str(docx_path)], check=True)

        with zipfile.ZipFile(docx_path) as zf:
            xml = zf.read("word/document.xml").decode("utf-8")

    paras = re.findall(r"<w:p\b.*?</w:p>", xml, flags=re.S)
    math_paras = [p for p in paras if "<m:oMathPara" in p]
    if len(math_paras) != len(FORMULAS):
        raise RuntimeError(f"Expected {len(FORMULAS)} OMML formulas, found {len(math_paras)}")

    fixed = []
    for para in math_paras:
        omml = re.search(r"<m:oMathPara\b.*?</m:oMathPara>", para, flags=re.S)
        if not omml:
            raise RuntimeError("Pandoc paragraph did not contain m:oMathPara")
        normalized_omml = normalize_pandoc_omml(omml.group(0))
        fixed.append(
            '<w:p><w:pPr><w:spacing w:before="80" w:after="120" w:line="300" w:lineRule="auto"/>'
            '<w:jc w:val="center"/>'
            f"</w:pPr>{normalized_omml}</w:p>"
        )
    return fixed


def normalize_pandoc_omml(xml: str) -> str:
    """Adjust Pandoc OMML into the stricter element order used by validator."""

    # Pandoc emits delimiter properties as beg/end/sep/grow. The OpenXML
    # schema order expected by the local validator is beg/sep/end/grow.
    xml = re.sub(
        r"<m:dPr><m:begChr(?P<beg>[^>]*)/><m:endChr(?P<end>[^>]*)/><m:sepChr(?P<sep>[^>]*)/><m:grow\s*/></m:dPr>",
        lambda m: (
            f"<m:dPr><m:begChr{m.group('beg')}/>"
            f"<m:sepChr{m.group('sep')}/>"
            f"<m:endChr{m.group('end')}/><m:grow /></m:dPr>"
        ),
        xml,
    )

    # Pandoc can write math run properties as sty/scr. The strict order is
    # scr before sty.
    xml = re.sub(
        r"<m:rPr><m:sty(?P<sty>[^>]*)/><m:scr(?P<scr>[^>]*)/></m:rPr>",
        lambda m: f"<m:rPr><m:scr{m.group('scr')}/><m:sty{m.group('sty')}/></m:rPr>",
        xml,
    )

    return xml


def replace_formula_paragraphs(document_xml: str, replacements: list[str]) -> tuple[str, list[dict[str, str]]]:
    updated = document_xml
    replacement_log = []
    for (plain, latex), replacement in zip(FORMULAS, replacements, strict=True):
        escaped = html.escape(plain, quote=False)
        pattern = re.compile(rf"<w:p\b(?:(?!</w:p>).)*?{re.escape(escaped)}(?:(?!</w:p>).)*?</w:p>", re.S)
        updated, count = pattern.subn(replacement, updated, count=1)
        if count != 1:
            raise RuntimeError(f"Expected to replace one paragraph for: {plain!r}; replaced {count}")
        replacement_log.append({"plain": plain, "latex": latex})
    return updated, replacement_log


def write_docx_with_document_xml(input_docx: Path, output_docx: Path, document_xml: str) -> None:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        unpacked = tmp / "docx"
        unpacked.mkdir()
        with zipfile.ZipFile(input_docx) as zf:
            zf.extractall(unpacked)

        (unpacked / "word" / "document.xml").write_text(document_xml, encoding="utf-8")

        with zipfile.ZipFile(output_docx, "w", zipfile.ZIP_DEFLATED) as zf:
            for file in sorted(unpacked.rglob("*")):
                if file.is_file():
                    zf.write(file, file.relative_to(unpacked).as_posix())


def main() -> None:
    if not INPUT.exists():
        raise FileNotFoundError(INPUT)
    if shutil.which("pandoc") is None:
        raise RuntimeError("pandoc is required to generate OMML formulas")

    with zipfile.ZipFile(INPUT) as zf:
        document_xml = zf.read("word/document.xml").decode("utf-8")

    omml_paras = make_omml_paragraphs()
    updated_xml, replacement_log = replace_formula_paragraphs(document_xml, omml_paras)
    write_docx_with_document_xml(INPUT, OUTPUT, updated_xml)

    MANIFEST.write_text(
        json.dumps(
            {
                "output": OUTPUT.name,
                "source": INPUT.name,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "change": "Replaced ten linear text formulas with Word OMML formulas generated from LaTeX via Pandoc.",
                "formula_count": len(replacement_log),
                "replacements": replacement_log,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {OUTPUT}")
    print(f"Wrote {MANIFEST}")


if __name__ == "__main__":
    main()
