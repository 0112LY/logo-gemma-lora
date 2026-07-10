"""Create the deterministic training split used by the LoRA experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


LOW_INFORMATION_PROMPTS = {"placeholder", "todo", "tbd", "n/a"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/train.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/train_filtered.jsonl"))
    args = parser.parse_args()

    kept: list[str] = []
    excluded: list[dict[str, object]] = []
    for row_number, line in enumerate(args.input.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        messages = row.get("messages", [])
        if [message.get("role") for message in messages] != ["system", "user", "assistant"]:
            raise ValueError(f"row {row_number}: unexpected chat structure")
        prompt = messages[1]["content"].strip()
        if len(prompt) < 100 or prompt.lower() in LOW_INFORMATION_PROMPTS:
            excluded.append({"row": row_number, "prompt": prompt})
            continue
        kept.append(json.dumps(row, ensure_ascii=False, separators=(",", ":")))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(kept) + "\n", encoding="utf-8")
    print(f"input={len(kept) + len(excluded)} kept={len(kept)} excluded={len(excluded)}")
    for item in excluded:
        print(f"excluded row {item['row']}: {item['prompt']!r}")


if __name__ == "__main__":
    main()
