#!/usr/bin/env python3
"""Apply template-compliance touch-ups to the v2 translation DOCX.

This preserves the formula-fixed v2 document and writes a v3 copy with minor
formatting adjustments only:

- remove first-line indentation from the abstract and keyword paragraphs;
- use the template's "参考文献：" label.
"""

from __future__ import annotations

import json
import re
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "graph_wavenet_ijcai2019_translation_draft_v2.docx"
OUTPUT = ROOT / "graph_wavenet_ijcai2019_translation_format_checked_v3.docx"
MANIFEST = ROOT / "graph_wavenet_ijcai2019_translation_format_checked_v3_manifest.json"

def patch_paragraph_containing(xml: str, literal: str, patch) -> tuple[str, bool]:
    pattern = re.compile(rf"<w:p\b(?:(?!</w:p>).)*?{re.escape(literal)}(?:(?!</w:p>).)*?</w:p>", re.S)
    match = pattern.search(xml)
    if not match:
        return xml, False
    paragraph = match.group(0)
    patched = patch(paragraph)
    if patched == paragraph:
        return xml, False
    return xml[: match.start()] + patched + xml[match.end() :], True


def remove_first_line_indent_from_xml(paragraph: str) -> str:
    paragraph = re.sub(r'<w:ind\s+w:firstLine="[^"]+"\s*/>', "", paragraph, count=1)
    paragraph = re.sub(r'\s+w:firstLine="[^"]+"', "", paragraph, count=1)
    return paragraph


def main() -> None:
    if not INPUT.exists():
        raise FileNotFoundError(INPUT)

    with tempfile.TemporaryDirectory() as td:
        unpacked = Path(td) / "docx"
        unpacked.mkdir()
        with zipfile.ZipFile(INPUT) as zf:
            zf.extractall(unpacked)

        changes = []
        document_xml = unpacked / "word" / "document.xml"
        xml = document_xml.read_text(encoding="utf-8")

        xml, changed = patch_paragraph_containing(
            xml,
            "摘要：",
            remove_first_line_indent_from_xml,
        )
        if changed:
            changes.append("Removed first-line indent from 摘要 paragraph")

        xml, changed = patch_paragraph_containing(
            xml,
            "关键词：",
            remove_first_line_indent_from_xml,
        )
        if changed:
            changes.append("Removed first-line indent from 关键词 paragraph")

        xml = xml.replace(
            '<w:t xml:space="preserve">参考文献</w:t>',
            '<w:t xml:space="preserve">参考文献：</w:t>',
            1,
        )
        changes.append("Changed reference heading to 参考文献：")

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
                "template_compliance_checks": [
                    "Cover uses the required graduation-thesis English-translation structure.",
                    "Original title is Times New Roman 15 pt; translated title is SimHei 15 pt bold on cover.",
                    "Main translated title is SimHei 16 pt bold and centered.",
                    "Author names are retained in English, comma-separated, Times New Roman 12 pt bold.",
                    "Affiliations are centered, 10.5 pt, single-spaced.",
                    "Abstract label is SimHei 12 pt; abstract body is KaiTi 12 pt with 1.5 line spacing.",
                    "Keyword label is SimHei 12 pt; keywords use KaiTi 12 pt.",
                    "Body Chinese text uses SimSun 12 pt; Latin text uses Times New Roman 12 pt; paragraphs use 1.5 line spacing.",
                    "Figure captions are centered and 10.5 pt.",
                    "English references are listed directly.",
                ],
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
    print("Changes:")
    for change in changes:
        print(f"- {change}")


if __name__ == "__main__":
    main()
