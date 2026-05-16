#!/usr/bin/env python3
"""Replace formula placeholders in the generated thesis DOCX with Word OMML."""

from __future__ import annotations

import argparse
import html
import json
import re
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path


def make_omml_runs(formulas: list[dict[str, object]]) -> dict[str, str]:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        markdown = "\n\n".join(f"$$\n{item['latex']}\n$$" for item in formulas)
        md_path = tmp / "formulas.md"
        docx_path = tmp / "formulas.docx"
        md_path.write_text(markdown, encoding="utf-8")
        subprocess.run(["pandoc", str(md_path), "-o", str(docx_path)], check=True)
        with zipfile.ZipFile(docx_path) as zf:
            xml = zf.read("word/document.xml").decode("utf-8")

    paras = re.findall(r"<w:p\b.*?</w:p>", xml, flags=re.S)
    math_paras = [p for p in paras if "<m:oMathPara" in p]
    if len(math_paras) != len(formulas):
        raise RuntimeError(f"Expected {len(formulas)} OMML formulas, found {len(math_paras)}")

    replacements: dict[str, str] = {}
    for item, para in zip(formulas, math_paras, strict=True):
        match = re.search(r"<m:oMath\b.*?</m:oMath>", para, flags=re.S)
        if not match:
            raise RuntimeError(f"Formula {item['placeholder']} did not produce m:oMath")
        replacements[str(item["placeholder"])] = normalize_pandoc_omml(match.group(0))
    return replacements


def normalize_pandoc_omml(xml: str) -> str:
    xml = re.sub(
        r"<m:dPr><m:begChr(?P<beg>[^>]*)/><m:endChr(?P<end>[^>]*)/><m:sepChr(?P<sep>[^>]*)/><m:grow\s*/></m:dPr>",
        lambda m: (
            f"<m:dPr><m:begChr{m.group('beg')}/>"
            f"<m:sepChr{m.group('sep')}/>"
            f"<m:endChr{m.group('end')}/><m:grow /></m:dPr>"
        ),
        xml,
    )
    xml = re.sub(
        r"<m:rPr><m:sty(?P<sty>[^>]*)/><m:scr(?P<scr>[^>]*)/></m:rPr>",
        lambda m: f"<m:rPr><m:scr{m.group('scr')}/><m:sty{m.group('sty')}/></m:rPr>",
        xml,
    )
    return xml


def replace_placeholders(document_xml: str, replacements: dict[str, str]) -> str:
    updated = document_xml
    for placeholder, omml in replacements.items():
        escaped = html.escape(placeholder, quote=False)
        pattern = re.compile(
            rf"<w:r\b(?:(?!</w:r>).)*?<w:t\b[^>]*>{re.escape(escaped)}</w:t>(?:(?!</w:r>).)*?</w:r>",
            re.S,
        )
        updated, count = pattern.subn(omml, updated, count=1)
        if count != 1:
            raise RuntimeError(f"Expected one placeholder run for {placeholder}, replaced {count}")
    return updated


def rewrite_docx(docx_path: Path, document_xml: str) -> None:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        unpacked = tmp / "docx"
        unpacked.mkdir()
        with zipfile.ZipFile(docx_path) as zf:
            zf.extractall(unpacked)
        (unpacked / "word" / "document.xml").write_text(document_xml, encoding="utf-8")
        tmp_docx = tmp / docx_path.name
        with zipfile.ZipFile(tmp_docx, "w", zipfile.ZIP_DEFLATED) as zf:
            for file in sorted(unpacked.rglob("*")):
                if file.is_file():
                    zf.write(file, file.relative_to(unpacked).as_posix())
        shutil.move(tmp_docx, docx_path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--docx", required=True)
    parser.add_argument("--formulas", required=True)
    args = parser.parse_args()
    docx_path = Path(args.docx)
    formulas = json.loads(Path(args.formulas).read_text(encoding="utf-8"))
    replacements = make_omml_runs(formulas)
    with zipfile.ZipFile(docx_path) as zf:
        document_xml = zf.read("word/document.xml").decode("utf-8")
    updated = replace_placeholders(document_xml, replacements)
    rewrite_docx(docx_path, updated)
    print(f"Replaced {len(replacements)} formulas in {docx_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
