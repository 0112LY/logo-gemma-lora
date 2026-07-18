"""从保存的真实推理记录生成报告用 SVG 对比面板。"""

from __future__ import annotations

import copy
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "examples" / "final_visible_4.jsonl"
OUTPUT = ROOT / "examples"
SAMPLE_INDICES = (0, 5, 11, 15)
SVG_NS = "http://www.w3.org/2000/svg"
ET.register_namespace("", SVG_NS)


def q(name: str) -> str:
    return f"{{{SVG_NS}}}{name}"


def add_text(root: ET.Element, x: int, y: int, value: str, **attrs: str) -> None:
    defaults = {
        "x": str(x),
        "y": str(y),
        "font-family": "Arial, Microsoft YaHei, sans-serif",
        "fill": "#172033",
    }
    defaults.update(attrs)
    ET.SubElement(root, q("text"), defaults).text = value


def extract_svg(raw: str) -> str:
    match = re.search(r"<svg\b[\s\S]*?</svg\s*>", raw, flags=re.IGNORECASE)
    return match.group(0) if match else raw.strip()


def nested_svg(raw: str, x: int) -> ET.Element:
    group = ET.Element(q("g"))
    ET.SubElement(
        group,
        q("rect"),
        {
            "x": str(x), "y": "62", "width": "300", "height": "300", "rx": "12",
            "fill": "#ffffff", "stroke": "#ccd4e0", "stroke-width": "2",
        },
    )
    try:
        parsed = ET.fromstring(extract_svg(raw))
    except ET.ParseError:
        add_text(group, x + 150, 205, "XML 非法，无法渲染", **{
            "text-anchor": "middle", "font-size": "20", "fill": "#b42318",
        })
        return group

    parsed = copy.deepcopy(parsed)
    parsed.set("x", str(x + 12))
    parsed.set("y", "74")
    parsed.set("width", "276")
    parsed.set("height", "276")
    parsed.set("preserveAspectRatio", "xMidYMid meet")
    if "viewBox" not in parsed.attrib:
        parsed.set("viewBox", "0 0 256 256")
    group.append(parsed)
    return group


def build(record: dict, sample_index: int) -> None:
    root = ET.Element(q("svg"), {"viewBox": "0 0 760 430", "width": "760", "height": "430"})
    ET.SubElement(root, q("rect"), {"width": "760", "height": "430", "fill": "#f6f8fb"})
    add_text(root, 30, 34, f"验证集样本 {sample_index}", **{"font-size": "20", "font-weight": "700"})
    add_text(root, 200, 55, "数据集参考徽标", **{"text-anchor": "middle", "font-size": "16", "font-weight": "600"})
    add_text(root, 560, 55, "最终提交模型的真实输出", **{"text-anchor": "middle", "font-size": "16", "font-weight": "600"})
    root.append(nested_svg(record["labels"], 50))
    root.append(nested_svg(record["response"], 410))
    add_text(root, 380, 400, "右图未经人工修复；完整闭合不等于具备有效视觉语义。", **{
        "text-anchor": "middle", "font-size": "15", "fill": "#475467",
    })
    ET.ElementTree(root).write(
        OUTPUT / f"final_comparison_{sample_index:02d}.svg",
        encoding="utf-8",
        xml_declaration=True,
    )


def main() -> None:
    records = [json.loads(line) for line in SOURCE.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(records) != len(SAMPLE_INDICES):
        raise ValueError(f"expected {len(SAMPLE_INDICES)} records, got {len(records)}")
    OUTPUT.mkdir(exist_ok=True)
    for sample_index, record in zip(SAMPLE_INDICES, records):
        build(record, sample_index)
    print(f"generated {len(records)} panels in {OUTPUT}")


if __name__ == "__main__":
    main()
