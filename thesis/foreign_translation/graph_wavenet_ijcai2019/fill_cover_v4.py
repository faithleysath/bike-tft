#!/usr/bin/env python3
"""Fill student information on the translation cover.

Input is the format-checked v3 DOCX. Output is a new v4 DOCX so earlier
versions remain available for reconstruction.
"""

from __future__ import annotations

import json
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "graph_wavenet_ijcai2019_translation_format_checked_v3.docx"
OUTPUT = ROOT / "graph_wavenet_ijcai2019_translation_final_v4.docx"
MANIFEST = ROOT / "graph_wavenet_ijcai2019_translation_final_v4_manifest.json"

FIELDS = {
    "学生姓名：________________": "学生姓名：吴天一",
    "学    号：________________": "学    号：202283250010",
    "专    业：________________": "专    业：数据科学与大数据技术",
    "学    院：________________": "学    院：未来技术学院",
    "指导教师：________________": "指导教师：胡伟",
}


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

        replacements = []
        for old, new in FIELDS.items():
            count = xml.count(old)
            if count != 1:
                raise RuntimeError(f"Expected exactly one occurrence of {old!r}, found {count}")
            xml = xml.replace(old, new, 1)
            replacements.append({"from": old, "to": new})

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
                "change": "Filled cover student information.",
                "student": {
                    "name": "吴天一",
                    "student_id": "202283250010",
                    "major": "数据科学与大数据技术",
                    "college": "未来技术学院",
                    "advisor": "胡伟",
                },
                "replacements": replacements,
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
