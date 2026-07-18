"""从本地详细评估记录生成报告用的真实前后对比 SVG 面板。"""

from __future__ import annotations

import copy
import json
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results_detailed.json"
OUTPUT = ROOT / "examples"
SELECTED = (1, 8, 10)
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


def nested_svg(raw: str, x: int, status: str) -> ET.Element:
    group = ET.Element(q("g"))
    ET.SubElement(
        group,
        q("rect"),
        {
            "x": str(x),
            "y": "54",
            "width": "240",
            "height": "240",
            "rx": "10",
            "fill": "#ffffff",
            "stroke": "#ccd4e0",
            "stroke-width": "2",
        },
    )
    try:
        parsed = ET.fromstring(raw)
    except ET.ParseError:
        add_text(group, x + 120, 165, "XML 非法", **{"text-anchor": "middle", "font-size": "22", "fill": "#b42318"})
        add_text(group, x + 120, 197, "无法渲染", **{"text-anchor": "middle", "font-size": "16", "fill": "#b42318"})
        return group

    parsed = copy.deepcopy(parsed)
    parsed.set("x", str(x + 8))
    parsed.set("y", "62")
    parsed.set("width", "224")
    parsed.set("height", "224")
    if "viewBox" not in parsed.attrib:
        parsed.set("viewBox", "0 0 256 256")
    group.append(parsed)
    if len(parsed) == 0:
        add_text(group, x + 120, 175, "空 SVG", **{"text-anchor": "middle", "font-size": "22", "fill": "#667085"})
        add_text(group, x + 120, 205, "无绘图元素", **{"text-anchor": "middle", "font-size": "15", "fill": "#667085"})
    return group


def build(sample: dict) -> None:
    index = int(sample["index"])
    root = ET.Element(q("svg"), {"viewBox": "0 0 900 350", "width": "900", "height": "350"})
    ET.SubElement(root, q("rect"), {"width": "900", "height": "350", "fill": "#f6f8fb"})
    add_text(root, 30, 32, f"示例 {index}", **{"font-size": "20", "font-weight": "700"})
    labels = ("数据集参考徽标", "微调前（BF16 基座）", "最终提交（零增量）")
    xs = (30, 330, 630)
    for x, label in zip(xs, labels):
        add_text(root, x + 120, 48, label, **{"text-anchor": "middle", "font-size": "15", "font-weight": "600"})

    root.append(nested_svg(sample["reference_svg"], xs[0], "参考"))
    root.append(nested_svg(sample["raw_output"], xs[1], "基座"))
    # 最终 checkpoint-1 的所有 LoRA B 矩阵为零；在同一推理路径下与基座严格等价。
    root.append(nested_svg(sample["raw_output"], xs[2], "最终"))
    status = "最终 adapter 的 LoRA 增量为 0，因此参数层面的前后输出相同。"
    add_text(root, 450, 330, status, **{"text-anchor": "middle", "font-size": "15", "fill": "#475467"})
    ET.ElementTree(root).write(OUTPUT / f"comparison_{index:02d}.svg", encoding="utf-8", xml_declaration=True)


def main() -> None:
    payload = json.loads(SOURCE.read_text(encoding="utf-8"))
    samples = {int(sample["index"]): sample for sample in payload["samples"]}
    OUTPUT.mkdir(exist_ok=True)
    for index in SELECTED:
        build(samples[index])
    print(f"generated {len(SELECTED)} panels in {OUTPUT}")


if __name__ == "__main__":
    main()

