"""Programmatic proxy reward for prompt-to-SVG generation.

The scorer deliberately favors robust, explainable checks over complete SVG
rendering. It is a training proxy, not a substitute for visual evaluation.
"""

from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable


WEIGHTS = {
    "validity": 0.30,
    "structure": 0.15,
    "canvas": 0.15,
    "visibility": 0.15,
    "simplicity": 0.10,
    "fidelity": 0.10,
    "safety": 0.05,
}

DRAWING_TAGS = {"circle", "ellipse", "line", "path", "polygon", "polyline", "rect", "text", "use"}
DANGEROUS_TAGS = {"embed", "foreignobject", "iframe", "object", "script"}
CONTAINER_TAGS = {"a", "defs", "g", "marker", "mask", "pattern", "symbol"}
NON_RENDERING_ANCESTORS = {"clippath", "defs", "marker", "mask", "pattern", "symbol"}
ALLOWED_TAGS = DRAWING_TAGS | CONTAINER_TAGS | {
    "clippath",
    "desc",
    "filter",
    "lineargradient",
    "metadata",
    "radialgradient",
    "stop",
    "style",
    "svg",
    "title",
    "use",
}
PATH_COMMAND_RE = re.compile(r"[MmLlHhVvCcSsQqTtAaZz]")
PATH_TOKEN_RE = re.compile(
    r"[MmLlHhVvCcSsQqTtAaZz]|[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|[,\s]+"
)
NUMBER_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")
SVG_RE = re.compile(r"<svg\b[^>]*>.*?</svg\s*>", re.IGNORECASE | re.DOTALL)
CODE_FENCE_RE = re.compile(r"```(?:svg|xml)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
URL_RE = re.compile(r"url\(\s*(['\"]?)(.*?)\1\s*\)", re.IGNORECASE)

COLOR_WORDS = {
    "black": ("#000", "#000000", "black", "rgb(0,0,0)"),
    "white": ("#fff", "#ffffff", "white", "rgb(255,255,255)"),
    "red": ("red", "#f00", "#ff0000"),
    "orange": ("orange", "#ffa500"),
    "yellow": ("yellow", "#ff0", "#ffff00"),
    "green": ("green", "#008000", "#0f0", "#00ff00"),
    "blue": ("blue", "#00f", "#0000ff"),
    "purple": ("purple", "#800080"),
    "pink": ("pink", "#ffc0cb"),
    "gray": ("gray", "grey", "#808080"),
    "grey": ("gray", "grey", "#808080"),
    "gold": ("gold", "#ffd700"),
    "银": ("silver", "#c0c0c0"),
    "黑": ("black", "#000", "#000000"),
    "白": ("white", "#fff", "#ffffff"),
    "红": ("red", "#f00", "#ff0000"),
    "橙": ("orange", "#ffa500"),
    "黄": ("yellow", "#ff0", "#ffff00"),
    "绿": ("green", "#008000", "#0f0", "#00ff00"),
    "蓝": ("blue", "#00f", "#0000ff"),
    "紫": ("purple", "#800080"),
    "粉": ("pink", "#ffc0cb"),
    "灰": ("gray", "grey", "#808080"),
    "金": ("gold", "#ffd700"),
}
SHAPE_WORDS = {
    "circle": {"circle"},
    "circular": {"circle"},
    "圆": {"circle", "ellipse"},
    "ellipse": {"ellipse"},
    "椭圆": {"ellipse"},
    "rectangle": {"rect"},
    "rectangular": {"rect"},
    "square": {"rect"},
    "矩形": {"rect"},
    "方形": {"rect"},
    "line": {"line", "polyline", "path"},
    "线": {"line", "polyline", "path"},
    "triangle": {"polygon", "path"},
    "三角": {"polygon", "path"},
    "polygon": {"polygon"},
    "多边形": {"polygon"},
    "text": {"text"},
    "letter": {"text"},
    "文字": {"text"},
    "字母": {"text"},
}


@dataclass
class RewardResult:
    total: float = 0.0
    validity: float = 0.0
    structure: float = 0.0
    canvas: float = 0.0
    visibility: float = 0.0
    simplicity: float = 0.0
    fidelity: float = 0.0
    safety: float = 0.0
    extracted_svg: str | None = None
    valid_xml: bool = False
    visible_elements: int = 0
    drawing_elements: int = 0
    reasons: list[str] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["total"] = round(self.total, 6)
        for name in WEIGHTS:
            value[name] = round(float(value[name]), 6)
        return value


def _local_name(name: str) -> str:
    return name.rsplit("}", 1)[-1].lower()


def extract_svg(output: str) -> str | None:
    """Extract the first complete SVG, tolerating a Markdown code fence."""
    if not isinstance(output, str):
        return None
    candidates = [match.group(1) for match in CODE_FENCE_RE.finditer(output)] + [output]
    for candidate in candidates:
        match = SVG_RE.search(candidate)
        if match:
            return match.group(0).strip()
    return None


def _parse_number(value: str | None) -> float | None:
    if value is None:
        return None
    match = NUMBER_RE.fullmatch(value.strip())
    if not match:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _parse_viewbox(root: ET.Element) -> tuple[float, float, float, float] | None:
    raw = root.attrib.get("viewBox")
    if not raw:
        return None
    parts = [part for part in re.split(r"[\s,]+", raw.strip()) if part]
    if len(parts) != 4:
        return None
    try:
        values = tuple(float(part) for part in parts)
    except ValueError:
        return None
    if not all(math.isfinite(part) for part in values) or values[2] <= 0 or values[3] <= 0:
        return None
    return values  # type: ignore[return-value]


def _valid_canvas(root: ET.Element) -> tuple[bool, tuple[float, float, float, float] | None]:
    viewbox = _parse_viewbox(root)
    if viewbox:
        return True, viewbox
    width = _parse_number(root.attrib.get("width"))
    height = _parse_number(root.attrib.get("height"))
    if width and height and width > 0 and height > 0:
        return True, (0.0, 0.0, width, height)
    return False, None


def _style_map(element: ET.Element) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in element.attrib.get("style", "").split(";"):
        if ":" in item:
            key, value = item.split(":", 1)
            result[key.strip().lower()] = value.strip().lower()
    return result


def _property(element: ET.Element, name: str) -> str | None:
    style = _style_map(element)
    return style.get(name, element.attrib.get(name, "")).strip().lower() or None


def _hidden(element: ET.Element) -> bool:
    if _property(element, "display") == "none" or _property(element, "visibility") == "hidden":
        return True
    opacity = _parse_number(_property(element, "opacity"))
    return opacity is not None and opacity <= 0


def _paint_is_visible(value: str | None) -> bool:
    if value is None:
        return False
    compact = re.sub(r"\s+", "", value.lower())
    if compact in {"none", "transparent", "rgba(0,0,0,0)"}:
        return False
    if compact.startswith("rgba(") and compact.endswith(",0)"):
        return False
    return True


def _visible_drawing(element: ET.Element) -> bool:
    tag = _local_name(element.tag)
    if tag not in DRAWING_TAGS or _hidden(element):
        return False
    if tag == "use":
        href = element.attrib.get("href") or element.attrib.get("{http://www.w3.org/1999/xlink}href")
        return bool(href and href.startswith("#"))
    fill = _property(element, "fill")
    stroke = _property(element, "stroke")
    fill_opacity = _parse_number(_property(element, "fill-opacity"))
    stroke_opacity = _parse_number(_property(element, "stroke-opacity"))
    visible_fill = _paint_is_visible(fill) and fill_opacity != 0
    visible_stroke = _paint_is_visible(stroke) and stroke_opacity != 0
    # SVG defaults filled shapes and text to black; line/polyline require stroke.
    if fill is None and tag not in {"line", "polyline"}:
        visible_fill = True
    if tag == "path" and not element.attrib.get("d", "").strip():
        return False
    if tag == "text" and not "".join(element.itertext()).strip():
        return False
    return visible_fill or visible_stroke


def _dangerous_content(root: ET.Element, svg: str) -> list[str]:
    findings: list[str] = []
    lowered = svg.lower()
    if "<!doctype" in lowered or "<!entity" in lowered:
        findings.append("DOCTYPE/entity declarations are forbidden")
    for element in root.iter():
        tag = _local_name(element.tag)
        if tag in DANGEROUS_TAGS:
            findings.append(f"dangerous tag: {tag}")
        if tag == "style":
            css = "".join(element.itertext())
            if "@import" in css.lower():
                findings.append("external CSS import")
            for match in URL_RE.finditer(css):
                target = match.group(2).strip()
                if target and not target.startswith("#"):
                    findings.append("external CSS url() reference")
        for raw_name, raw_value in element.attrib.items():
            name = _local_name(raw_name)
            value = raw_value.strip()
            if name.startswith("on"):
                findings.append(f"event handler attribute: {name}")
            if name in {"href", "src"} and value and not value.startswith("#"):
                findings.append(f"external reference in {name}")
            for match in URL_RE.finditer(value):
                target = match.group(2).strip()
                if target and not target.startswith("#"):
                    findings.append("external url() reference")
    return sorted(set(findings))


def _path_is_sane(path: ET.Element, coordinate_limit: float) -> tuple[bool, str | None]:
    data = path.attrib.get("d", "").strip()
    if not data:
        return False, "empty path data"
    if not PATH_COMMAND_RE.search(data):
        return False, "path has no recognized command"
    remainder = PATH_TOKEN_RE.sub("", data)
    if remainder:
        return False, "path contains invalid tokens"
    numbers = [float(value) for value in NUMBER_RE.findall(data)]
    if not all(math.isfinite(value) for value in numbers):
        return False, "path contains non-finite numbers"
    if any(abs(value) > coordinate_limit for value in numbers):
        return False, "path contains extreme coordinates"
    return True, None


def _numeric_geometry_score(
    elements: Iterable[ET.Element], canvas: tuple[float, float, float, float] | None
) -> tuple[float, list[str]]:
    checked = invalid = 0
    reasons: list[str] = []
    scale = max(canvas[2], canvas[3]) if canvas else 256.0
    extreme_limit = max(100_000.0, scale * 100.0)
    nonnegative = {"height", "r", "rx", "ry", "width"}
    numeric_attributes = {
        "circle": ("cx", "cy", "r"),
        "ellipse": ("cx", "cy", "rx", "ry"),
        "line": ("x1", "x2", "y1", "y2"),
        "rect": ("height", "rx", "ry", "width", "x", "y"),
        "text": ("x", "y"),
    }
    for element in elements:
        tag = _local_name(element.tag)
        if tag == "path":
            checked += 1
            sane, reason = _path_is_sane(element, extreme_limit)
            if not sane:
                invalid += 1
                reasons.append(reason or "invalid path")
            continue
        if tag in {"polygon", "polyline"}:
            checked += 1
            raw = element.attrib.get("points", "")
            values = [float(item) for item in NUMBER_RE.findall(raw)]
            if len(values) < 4 or len(values) % 2 or not all(math.isfinite(item) for item in values):
                invalid += 1
                reasons.append(f"invalid {tag} points")
            elif any(abs(item) > extreme_limit for item in values):
                invalid += 1
                reasons.append(f"extreme {tag} coordinates")
            continue
        for name in numeric_attributes.get(tag, ()):
            if name not in element.attrib:
                continue
            checked += 1
            number = _parse_number(element.attrib[name])
            if number is None or abs(number) > extreme_limit or (name in nonnegative and number < 0):
                invalid += 1
                reasons.append(f"invalid {tag}.{name}")
    return (1.0 if not checked else max(0.0, 1.0 - invalid / checked)), sorted(set(reasons))


def _explicit_colors(root: ET.Element) -> set[str]:
    colors: set[str] = set()
    for element in root.iter():
        for name in ("fill", "stroke", "color", "stop-color"):
            value = _property(element, name)
            if value and _paint_is_visible(value) and not value.startswith("url("):
                colors.add(re.sub(r"\s+", "", value.lower()))
    return colors


def _requested_text(prompt: str) -> list[str]:
    requirements: list[str] = []
    patterns = (
        r"(?:text|word|lettering|letters?|initials?|文字|字样|字母)\s*(?:of|reading|saying|为|是|：|:)?\s*[\"'“‘]([^\"'”’]{1,30})[\"'”’]",
        r"[\"'“‘]([^\"'”’]{1,12})[\"'”’]\s*(?:text|word|lettering|文字|字样|字母)",
    )
    for pattern in patterns:
        requirements.extend(match.strip().lower() for match in re.findall(pattern, prompt, re.IGNORECASE))
    return requirements


def _keyword_present(prompt: str, keyword: str) -> bool:
    if keyword.isascii() and keyword.replace(" ", "").isalpha():
        return re.search(rf"\b{re.escape(keyword)}\b", prompt, re.IGNORECASE) is not None
    return keyword.lower() in prompt.lower()


def _fidelity_score(prompt: str, root: ET.Element, tags: Counter[str]) -> tuple[float, dict[str, Any]]:
    lowered = prompt.lower()
    checks: list[bool] = []
    details: dict[str, Any] = {"colors": [], "shapes": [], "text": []}
    actual_colors = _explicit_colors(root)

    for word, accepted in COLOR_WORDS.items():
        if _keyword_present(lowered, word):
            matched = any(re.sub(r"\s+", "", value) in actual_colors for value in accepted)
            checks.append(matched)
            details["colors"].append({"requested": word, "matched": matched})
    for word, accepted_tags in SHAPE_WORDS.items():
        if _keyword_present(lowered, word):
            matched = any(tags[tag] > 0 for tag in accepted_tags)
            checks.append(matched)
            details["shapes"].append({"requested": word, "matched": matched})
    visible_text = " ".join("".join(element.itertext()) for element in root.iter() if _local_name(element.tag) == "text")
    visible_text = visible_text.lower()
    for requested in _requested_text(prompt):
        matched = requested in visible_text
        checks.append(matched)
        details["text"].append({"requested": requested, "matched": matched})

    # Neutral score when the proxy cannot verify any explicit requirement.
    return (sum(checks) / len(checks) if checks else 0.5), details


def _degeneration_score(svg: str, drawing_elements: list[ET.Element]) -> tuple[float, list[str]]:
    reasons: list[str] = []
    score = 1.0
    if len(svg) > 50_000:
        score -= 0.5
        reasons.append("SVG output is excessively long")
    serializations = [ET.tostring(element, encoding="unicode") for element in drawing_elements]
    if serializations:
        most_common = Counter(serializations).most_common(1)[0][1]
        if most_common >= 8 and most_common / len(serializations) >= 0.4:
            score -= 0.6
            reasons.append("many drawing elements are exact repetitions")
    if re.search(r"(.{40,400})\1{4,}", svg, re.DOTALL):
        score -= 0.5
        reasons.append("large repeated output fragment detected")
    return max(0.0, score), reasons


def score_svg(prompt: str, output: str) -> dict[str, Any]:
    """Score one prompt/output pair and return component diagnostics."""
    result = RewardResult()
    svg = extract_svg(output)
    result.extracted_svg = svg
    if svg is None:
        result.reasons.append("no complete SVG found")
        return result.to_dict()

    if "<!doctype" in svg.lower() or "<!entity" in svg.lower():
        result.reasons.append("DOCTYPE/entity declarations are forbidden")
        return result.to_dict()

    try:
        root = ET.fromstring(svg)
    except ET.ParseError as exc:
        result.validity = 0.2
        result.total = min(0.1, WEIGHTS["validity"] * result.validity)
        result.reasons.append(f"XML parse error: {exc}")
        return result.to_dict()

    result.valid_xml = True
    root_is_svg = _local_name(root.tag) == "svg"
    if not root_is_svg:
        result.validity = 0.4
        result.total = min(0.1, WEIGHTS["validity"] * result.validity)
        result.reasons.append("root element is not svg")
        return result.to_dict()

    elements = list(root.iter())
    parents = {child: parent for parent in elements for child in parent}

    def is_directly_rendered(element: ET.Element) -> bool:
        parent = parents.get(element)
        while parent is not None:
            if _local_name(parent.tag) in NON_RENDERING_ANCESTORS:
                return False
            parent = parents.get(parent)
        return True

    tags = Counter(_local_name(element.tag) for element in elements)
    drawing_elements = [
        element for element in elements if _local_name(element.tag) in DRAWING_TAGS and is_directly_rendered(element)
    ]
    rendered_tags = Counter(_local_name(element.tag) for element in drawing_elements)
    visible_elements = [element for element in drawing_elements if _visible_drawing(element)]
    result.drawing_elements = len(drawing_elements)
    result.visible_elements = len(visible_elements)
    result.validity = 1.0

    dangerous = _dangerous_content(root, svg)
    unknown_tags = sorted(tag for tag in tags if tag not in ALLOWED_TAGS)
    result.safety = 0.0 if dangerous else 1.0
    if dangerous:
        result.reasons.extend(dangerous)
        result.diagnostics = {"tags": dict(tags), "unknown_tags": unknown_tags}
        return result.to_dict()

    canvas_valid, canvas = _valid_canvas(root)
    result.structure = 0.5 + 0.3 * float(canvas_valid) + 0.2 * float(not unknown_tags)
    if unknown_tags:
        result.reasons.append(f"unknown tags: {', '.join(unknown_tags)}")
    if not canvas_valid:
        result.reasons.append("missing or invalid viewBox/width/height")

    geometry_score, geometry_reasons = _numeric_geometry_score(drawing_elements, canvas)
    result.canvas = (0.4 * float(canvas_valid)) + (0.6 * geometry_score)
    result.reasons.extend(geometry_reasons)

    explicit_colors = _explicit_colors(root)
    visible_ratio = len(visible_elements) / len(drawing_elements) if drawing_elements else 0.0
    result.visibility = 0.7 * visible_ratio + 0.3 * float(bool(explicit_colors))
    if not drawing_elements:
        result.reasons.append("no drawing elements")
    elif not visible_elements:
        result.reasons.append("no visible drawing elements")
    if not explicit_colors:
        result.reasons.append("no explicit visible fill/stroke color")

    count = len(drawing_elements)
    if 1 <= count <= 40:
        result.simplicity = 1.0
    elif count <= 100:
        result.simplicity = max(0.4, 1.0 - (count - 40) / 100)
    elif count <= 200:
        result.simplicity = max(0.1, 0.4 - (count - 100) / 400)
    else:
        result.simplicity = 0.0
        result.reasons.append("excessive drawing element count")

    result.fidelity, fidelity_details = _fidelity_score(prompt, root, rendered_tags)
    degeneration, degeneration_reasons = _degeneration_score(svg, drawing_elements)
    result.safety *= degeneration
    result.reasons.extend(degeneration_reasons)

    result.total = sum(WEIGHTS[name] * float(getattr(result, name)) for name in WEIGHTS)
    if not visible_elements:
        result.total = min(result.total, 0.2)
    result.total = max(0.0, min(1.0, result.total))
    result.reasons = list(dict.fromkeys(result.reasons))
    result.diagnostics = {
        "tags": dict(tags),
        "unknown_tags": unknown_tags,
        "explicit_colors": sorted(explicit_colors),
        "canvas": canvas,
        "geometry_score": round(geometry_score, 6),
        "fidelity": fidelity_details,
        "weights": WEIGHTS,
    }
    return result.to_dict()


def reward(prompt: str, output: str) -> float:
    """Return only the scalar reward for simple training/evaluation callers."""
    return float(score_svg(prompt, output)["total"])


compute_reward = reward


def reward_fn(prompts: list[str], completions: list[str], **_: Any) -> list[float]:
    """Batch-compatible wrapper used by common reward/evaluation interfaces."""
    if len(prompts) != len(completions):
        raise ValueError("prompts and completions must have the same length")
    return [reward(prompt, completion) for prompt, completion in zip(prompts, completions)]
