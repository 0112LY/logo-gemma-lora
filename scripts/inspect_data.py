"""Audit the published prompt-to-SVG JSONL splits.

The script is intentionally read-only with respect to the source JSONL files.
It writes a deterministic JSON report that can be regenerated before training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from transformers import AutoTokenizer


EXPECTED_ROLES = ["system", "user", "assistant"]
COLOR_ATTRIBUTES = {"color", "fill", "flood-color", "lighting-color", "stop-color", "stroke"}
STYLE_COLOR_RE = re.compile(
    r"(?:^|;)\s*(?:color|fill|flood-color|lighting-color|stop-color|stroke)\s*:\s*([^;]+)",
    re.IGNORECASE,
)
CANDIDATE_MAX_LENGTHS = (1024, 1536, 2048, 2560, 3072, 3584, 4096)


def percentile(values: list[int], percent: int) -> int:
    """Return a deterministic nearest-rank percentile."""
    if not values:
        return 0
    ordered = sorted(values)
    index = max(0, (len(ordered) * percent + 99) // 100 - 1)
    return ordered[min(index, len(ordered) - 1)]


def distribution(values: list[int]) -> dict[str, float | int]:
    if not values:
        return {"min": 0, "p50": 0, "p90": 0, "p95": 0, "p99": 0, "max": 0, "mean": 0.0}
    return {
        "min": min(values),
        "p50": percentile(values, 50),
        "p90": percentile(values, 90),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
        "max": max(values),
        "mean": round(statistics.fmean(values), 2),
    }


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.strip().encode("utf-8")).hexdigest()


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].lower()


def load_jsonl(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                errors.append({"line": line_number, "error": "blank_line"})
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append({"line": line_number, "error": "invalid_json", "detail": str(exc)})
                continue
            if not isinstance(value, dict):
                errors.append({"line": line_number, "error": "row_is_not_object"})
                continue
            rows.append(value)
    return rows, errors


def analyze_svg(svg: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "valid_xml": False,
        "svg_root": False,
        "has_viewbox": False,
        "viewbox_0_0_256_256": False,
        "tags": Counter(),
        "colors": Counter(),
        "element_count": 0,
        "error": None,
    }
    try:
        root = ET.fromstring(svg)
    except ET.ParseError as exc:
        result["error"] = str(exc)
        return result

    result["valid_xml"] = True
    result["svg_root"] = local_name(root.tag) == "svg"
    viewbox = root.attrib.get("viewBox", "").strip()
    result["has_viewbox"] = bool(viewbox)
    if viewbox:
        try:
            values = [float(part) for part in re.split(r"[\s,]+", viewbox) if part]
            result["viewbox_0_0_256_256"] = values == [0.0, 0.0, 256.0, 256.0]
        except ValueError:
            pass

    for element in root.iter():
        tag = local_name(element.tag)
        result["tags"][tag] += 1
        result["element_count"] += 1
        for attribute, value in element.attrib.items():
            attribute = local_name(attribute)
            if attribute in COLOR_ATTRIBUTES:
                result["colors"][value.strip().lower()] += 1
            elif attribute == "style":
                for match in STYLE_COLOR_RE.finditer(value):
                    result["colors"][match.group(1).strip().lower()] += 1
    return result


def count_tokens(tokenizer: Any, messages: list[dict[str, str]]) -> tuple[int, int, int]:
    prompt = tokenizer.apply_chat_template(messages[:2], tokenize=False, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    prompt_tokens = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
    full_tokens = len(tokenizer(full, add_special_tokens=False)["input_ids"])
    assistant_tokens = len(tokenizer(messages[2]["content"], add_special_tokens=False)["input_ids"])
    return prompt_tokens, assistant_tokens, full_tokens


def duplicate_count(values: Iterable[str]) -> int:
    counts = Counter(values)
    return sum(count - 1 for count in counts.values() if count > 1)


def duplicate_groups(values: list[str]) -> list[dict[str, Any]]:
    groups: dict[str, list[int]] = {}
    original_values: dict[str, str] = {}
    for row_number, value in enumerate(values, start=1):
        digest = sha256_text(value)
        groups.setdefault(digest, []).append(row_number)
        original_values.setdefault(digest, value.strip())
    return [
        {
            "sha256": digest,
            "rows": rows,
            "preview": original_values[digest][:160],
        }
        for digest, rows in groups.items()
        if len(rows) > 1
    ]


def audit_split(path: Path, tokenizer: Any) -> tuple[dict[str, Any], dict[str, set[str]]]:
    rows, json_errors = load_jsonl(path)
    structural_errors: list[dict[str, Any]] = []
    svg_errors: list[dict[str, Any]] = []
    prompts: list[str] = []
    svgs: list[str] = []
    prompt_chars: list[int] = []
    svg_chars: list[int] = []
    prompt_tokens: list[int] = []
    assistant_tokens: list[int] = []
    full_tokens: list[int] = []
    element_counts: list[int] = []
    tags: Counter[str] = Counter()
    colors: Counter[str] = Counter()
    suspicious_prompts: list[dict[str, Any]] = []
    valid_xml = svg_roots = viewboxes = canonical_viewboxes = 0

    for row_index, row in enumerate(rows, start=1):
        messages = row.get("messages")
        if not isinstance(messages, list) or len(messages) != 3:
            structural_errors.append({"row": row_index, "error": "expected_three_messages"})
            continue
        if not all(isinstance(message, dict) for message in messages):
            structural_errors.append({"row": row_index, "error": "message_is_not_object"})
            continue
        roles = [message.get("role") for message in messages]
        if roles != EXPECTED_ROLES:
            structural_errors.append({"row": row_index, "error": "unexpected_roles", "roles": roles})
            continue
        contents = [message.get("content") for message in messages]
        if not all(isinstance(content, str) and content.strip() for content in contents):
            structural_errors.append({"row": row_index, "error": "empty_or_non_string_content"})
            continue

        prompt, svg = contents[1], contents[2]
        normalized_prompt = prompt.strip().lower()
        if len(prompt.strip()) < 100 or normalized_prompt in {"placeholder", "todo", "tbd", "n/a"}:
            suspicious_prompts.append(
                {"row": row_index, "characters": len(prompt.strip()), "preview": prompt.strip()[:160]}
            )
        prompts.append(prompt)
        svgs.append(svg)
        prompt_chars.append(len(prompt))
        svg_chars.append(len(svg))
        p_tokens, a_tokens, f_tokens = count_tokens(tokenizer, messages)
        prompt_tokens.append(p_tokens)
        assistant_tokens.append(a_tokens)
        full_tokens.append(f_tokens)

        svg_result = analyze_svg(svg)
        valid_xml += int(svg_result["valid_xml"])
        svg_roots += int(svg_result["svg_root"])
        viewboxes += int(svg_result["has_viewbox"])
        canonical_viewboxes += int(svg_result["viewbox_0_0_256_256"])
        element_counts.append(svg_result["element_count"])
        tags.update(svg_result["tags"])
        colors.update(svg_result["colors"])
        if not svg_result["valid_xml"] or not svg_result["svg_root"]:
            svg_errors.append({"row": row_index, "error": svg_result["error"] or "root_is_not_svg"})

    valid_rows = len(prompts)
    summary = {
        "path": path.as_posix(),
        "rows": len(rows),
        "valid_chat_rows": valid_rows,
        "json_errors": json_errors,
        "structural_errors": structural_errors,
        "svg_errors": svg_errors,
        "svg": {
            "valid_xml": valid_xml,
            "svg_root": svg_roots,
            "has_viewbox": viewboxes,
            "viewbox_0_0_256_256": canonical_viewboxes,
            "element_count": distribution(element_counts),
            "top_tags": tags.most_common(20),
            "top_colors": colors.most_common(20),
        },
        "lengths": {
            "prompt_characters": distribution(prompt_chars),
            "svg_characters": distribution(svg_chars),
            "prompt_tokens_with_chat_template": distribution(prompt_tokens),
            "assistant_svg_tokens": distribution(assistant_tokens),
            "full_sequence_tokens": distribution(full_tokens),
            "candidate_max_length_coverage": {
                str(limit): {
                    "rows_fitting": sum(value <= limit for value in full_tokens),
                    "rows_truncated": sum(value > limit for value in full_tokens),
                    "coverage_percent": round(100 * sum(value <= limit for value in full_tokens) / valid_rows, 2)
                    if valid_rows
                    else 0.0,
                }
                for limit in CANDIDATE_MAX_LENGTHS
            },
        },
        "duplicates": {
            "prompt_rows_beyond_first": duplicate_count(sha256_text(value) for value in prompts),
            "svg_rows_beyond_first": duplicate_count(sha256_text(value) for value in svgs),
            "prompt_groups": duplicate_groups(prompts),
            "svg_groups": duplicate_groups(svgs),
        },
        "suspicious_prompts": suspicious_prompts,
    }
    hashes = {
        "prompts": {sha256_text(value) for value in prompts},
        "svgs": {sha256_text(value) for value in svgs},
    }
    return summary, hashes


def recommend_max_length(max_full_tokens: int) -> int:
    """Round the observed maximum up to a practical multiple of 256."""
    return max(256, ((max_full_tokens + 255) // 256) * 256)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=Path("data/train.jsonl"))
    parser.add_argument("--valid", type=Path, default=Path("data/valid.jsonl"))
    parser.add_argument("--model", default="models/gemma-3-270m-it")
    parser.add_argument("--output", type=Path, default=Path("outputs/dataset_audit.json"))
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    train, train_hashes = audit_split(args.train, tokenizer)
    valid, valid_hashes = audit_split(args.valid, tokenizer)
    maximum = max(
        int(train["lengths"]["full_sequence_tokens"]["max"]),
        int(valid["lengths"]["full_sequence_tokens"]["max"]),
    )
    report = {
        "schema_version": 1,
        "tokenizer": args.model,
        "train": train,
        "valid": valid,
        "cross_split_overlap": {
            "prompts": len(train_hashes["prompts"] & valid_hashes["prompts"]),
            "svgs": len(train_hashes["svgs"] & valid_hashes["svgs"]),
        },
        "training_recommendation": {
            "observed_max_full_sequence_tokens": maximum,
            "minimum_no_truncation_max_length": recommend_max_length(maximum),
            "preferred_ai_studio_max_length": recommend_max_length(maximum),
            "local_2_gib_smoke_test_max_length": 2048,
            "note": "Use 2048 only for local smoke tests; use 3584 on AI Studio to retain every published sample.",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"train rows: {train['rows']} (valid chat: {train['valid_chat_rows']})")
    print(f"valid rows: {valid['rows']} (valid chat: {valid['valid_chat_rows']})")
    print(f"train/valid prompt overlap: {report['cross_split_overlap']['prompts']}")
    print(f"train/valid SVG overlap: {report['cross_split_overlap']['svgs']}")
    print(f"observed max sequence: {maximum} tokens")
    print(f"recommended no-truncation max_length: {report['training_recommendation']['minimum_no_truncation_max_length']}")
    print(f"report: {args.output}")


if __name__ == "__main__":
    main()
