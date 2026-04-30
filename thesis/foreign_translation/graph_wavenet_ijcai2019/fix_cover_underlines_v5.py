#!/usr/bin/env python3
"""Restore underlined cover fields while keeping filled student info.

Input is v4 with filled text. Output is v5, preserving prior versions.
"""

from __future__ import annotations

import html
import json
import re
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "graph_wavenet_ijcai2019_translation_final_v4.docx"
OUTPUT = ROOT / "graph_wavenet_ijcai2019_translation_final_v5.docx"
MANIFEST = ROOT / "graph_wavenet_ijcai2019_translation_final_v5_manifest.json"

FIELDS = [
    ("学生姓名：吴天一", "学生姓名：", "吴天一"),
    ("学    号：202283250010", "学    号：", "202283250010"),
    ("专    业：数据科学与大数据技术", "专    业：", "数据科学与大数据技术"),
    ("学    院：未来技术学院", "学    院：", "未来技术学院"),
    ("指导教师：胡伟", "指导教师：", "胡伟"),
]


def display_units(text: str) -> int:
    units = 0
    for ch in text:
        units += 2 if ord(ch) > 127 else 1
    return units


def padded_value(value: str, target_units: int = 24) -> str:
    # Normal spaces are underlined by Word when xml:space="preserve" is set.
    missing = max(4, target_units - display_units(value))
    left = missing // 2
    right = missing - left
    return " " * left + value + " " * right


def add_underline_to_rpr(rpr: str) -> str:
    if "<w:u " in rpr or "<w:u/" in rpr:
        return rpr
    return rpr.replace("</w:rPr>", '<w:u w:val="single"/></w:rPr>', 1)


def replace_cover_field(xml: str, full_text: str, label: str, value: str) -> tuple[str, bool]:
    escaped = html.escape(full_text, quote=False)
    paragraph_re = re.compile(rf"<w:p\b(?:(?!</w:p>).)*?{re.escape(escaped)}(?:(?!</w:p>).)*?</w:p>", re.S)
    match = paragraph_re.search(xml)
    if not match:
        return xml, False

    paragraph = match.group(0)
    run_re = re.compile(
        rf"(<w:r><w:rPr>.*?</w:rPr>)<w:t xml:space=\"preserve\">{re.escape(escaped)}</w:t></w:r>",
        re.S,
    )
    run_match = run_re.search(paragraph)
    if not run_match:
        raise RuntimeError(f"Could not find single text run for {full_text!r}")

    run_start = run_match.group(1)
    rpr = re.search(r"<w:rPr>.*?</w:rPr>", run_start, re.S)
    if not rpr:
        raise RuntimeError(f"Could not find run properties for {full_text!r}")

    plain_rpr = rpr.group(0)
    underlined_rpr = add_underline_to_rpr(plain_rpr)
    new_runs = (
        f"<w:r>{plain_rpr}<w:t xml:space=\"preserve\">{html.escape(label, quote=False)}</w:t></w:r>"
        f"<w:r>{underlined_rpr}<w:t xml:space=\"preserve\">{html.escape(padded_value(value), quote=False)}</w:t></w:r>"
    )
    new_paragraph = paragraph[: run_match.start()] + new_runs + paragraph[run_match.end() :]
    return xml[: match.start()] + new_paragraph + xml[match.end() :], True


def main() -> None:
    if not INPUT.exists():
        raise FileNotFoundError(INPUT)

    changes = []
    with tempfile.TemporaryDirectory() as td:
        unpacked = Path(td) / "docx"
        unpacked.mkdir()
        with zipfile.ZipFile(INPUT) as zf:
            zf.extractall(unpacked)

        document_xml = unpacked / "word" / "document.xml"
        xml = document_xml.read_text(encoding="utf-8")
        for full_text, label, value in FIELDS:
            xml, changed = replace_cover_field(xml, full_text, label, value)
            if not changed:
                raise RuntimeError(f"Failed to replace {full_text!r}")
            changes.append({"field": label, "value": value})

        document_xml.write_text(xml, encoding="utf-8")

        with zipfile.ZipFile(OUTPUT, "w", zipfile.ZIP_DEFLATED) as zf:
            for file in sorted(unpacked.rglob("*")):
                if file.is_file():
                    zf.write(file, file.relative_to(unpacked).as_posix())

    MANIFEST.write_text(
        json.dumps(
            {
                "output": OUTPUT.name,
                "source": INPUT.name,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "change": "Restored underlined cover-field values.",
                "changes": changes,
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
