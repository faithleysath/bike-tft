#!/usr/bin/env python3
"""Polish the cover information block using an invisible two-column table.

The v5 cover restored underlines but relied on centered paragraphs and padded
spaces, which made the filled fields visually uneven. This version replaces the
five field paragraphs with a centered table: labels are right-aligned in a
fixed-width column, values are centered in equal-width cells with only bottom
borders visible.
"""

from __future__ import annotations

import json
import re
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "graph_wavenet_ijcai2019_translation_final_v5.docx"
OUTPUT = ROOT / "graph_wavenet_ijcai2019_translation_final_v6.docx"
MANIFEST = ROOT / "graph_wavenet_ijcai2019_translation_final_v6_manifest.json"

FIELDS = [
    ("学生姓名：", "吴天一"),
    ("学    号：", "202283250010"),
    ("专    业：", "数据科学与大数据技术"),
    ("学    院：", "未来技术学院"),
    ("指导教师：", "胡伟"),
]


def run(text: str, *, size: int = 30, east_asia: str = "SimSun") -> str:
    return (
        "<w:r><w:rPr>"
        f'<w:rFonts w:ascii="Times New Roman" w:eastAsia="{east_asia}" w:hAnsi="Times New Roman"/>'
        f'<w:sz w:val="{size}"/><w:szCs w:val="{size}"/>'
        "</w:rPr>"
        f'<w:t xml:space="preserve">{text}</w:t>'
        "</w:r>"
    )


def paragraph(text: str, *, align: str, size: int = 30) -> str:
    return (
        "<w:p><w:pPr>"
        '<w:spacing w:before="0" w:after="0"/>'
        f'<w:jc w:val="{align}"/>'
        "</w:pPr>"
        f"{run(text, size=size)}"
        "</w:p>"
    )


def cell(text: str, *, width: int, align: str, bottom_border: bool = False) -> str:
    borders = ""
    if bottom_border:
        borders = (
            "<w:tcBorders>"
            '<w:bottom w:val="single" w:sz="8" w:space="0" w:color="000000"/>'
            "</w:tcBorders>"
        )
    return (
        "<w:tc>"
        "<w:tcPr>"
        f'<w:tcW w:w="{width}" w:type="dxa"/>'
        f"{borders}"
        '<w:tcMar><w:top w:w="45" w:type="dxa"/><w:left w:w="70" w:type="dxa"/>'
        '<w:bottom w:w="45" w:type="dxa"/><w:right w:w="70" w:type="dxa"/></w:tcMar>'
        '<w:vAlign w:val="center"/>'
        "</w:tcPr>"
        f"{paragraph(text, align=align)}"
        "</w:tc>"
    )


def cover_table() -> str:
    rows = []
    for label, value in FIELDS:
        rows.append(
            "<w:tr>"
            '<w:trPr><w:trHeight w:val="520" w:hRule="atLeast"/></w:trPr>'
            f'{cell(label, width=1900, align="right")}'
            f'{cell(value, width=3500, align="center", bottom_border=True)}'
            "</w:tr>"
        )

    return (
        "<w:tbl>"
        "<w:tblPr>"
        '<w:tblW w:w="5400" w:type="dxa"/>'
        '<w:jc w:val="center"/>'
        '<w:tblBorders><w:top w:val="nil"/><w:left w:val="nil"/><w:bottom w:val="nil"/>'
        '<w:right w:val="nil"/><w:insideH w:val="nil"/><w:insideV w:val="nil"/></w:tblBorders>'
        '<w:tblCellMar><w:top w:w="0" w:type="dxa"/><w:left w:w="0" w:type="dxa"/>'
        '<w:bottom w:w="0" w:type="dxa"/><w:right w:w="0" w:type="dxa"/></w:tblCellMar>'
        "</w:tblPr>"
        '<w:tblGrid><w:gridCol w:w="1900"/><w:gridCol w:w="3500"/></w:tblGrid>'
        + "".join(rows)
        + "</w:tbl>"
        '<w:p><w:pPr><w:spacing w:before="0" w:after="420"/></w:pPr></w:p>'
    )


def main() -> None:
    if not INPUT.exists():
        raise FileNotFoundError(INPUT)

    with tempfile.TemporaryDirectory() as td:
        unpacked = Path(td) / "docx"
        unpacked.mkdir()
        with zipfile.ZipFile(INPUT) as zf:
            zf.extractall(unpacked)

        document_xml = unpacked / "word" / "document.xml"
        xml = document_xml.read_text(encoding="utf-8")

        pattern = re.compile(
            r"<w:p\b(?:(?!</w:p>).)*?学生姓名：(?:(?!</w:p>).)*?</w:p>"
            r"<w:p\b(?:(?!</w:p>).)*?学\s*号：(?:(?!</w:p>).)*?</w:p>"
            r"<w:p\b(?:(?!</w:p>).)*?专\s*业：(?:(?!</w:p>).)*?</w:p>"
            r"<w:p\b(?:(?!</w:p>).)*?学\s*院：(?:(?!</w:p>).)*?</w:p>"
            r"<w:p\b(?:(?!</w:p>).)*?指导教师：(?:(?!</w:p>).)*?</w:p>",
            re.S,
        )
        replacement = cover_table()
        xml, count = pattern.subn(replacement, xml, count=1)
        if count != 1:
            raise RuntimeError(f"Expected to replace one cover information block, replaced {count}")

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
                "change": "Replaced centered cover-field paragraphs with an invisible aligned two-column table.",
                "fields": [{"label": label, "value": value} for label, value in FIELDS],
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
