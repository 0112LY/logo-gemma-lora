"""Generate validation SVGs and score them with the local proxy reward.

The detailed output is checkpointed after every sample so a slow local run can
be resumed. The compact result intentionally excludes raw generations.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from student_kit.reward import WEIGHTS, score_svg


SEED = 42
COMPONENTS = tuple(WEIGHTS)


class StopOnTokenSequence(StoppingCriteria):
    """Stop batch-size-one generation after a configured token suffix."""

    def __init__(self, token_sequences: list[list[int]]) -> None:
        self.token_sequences = [sequence for sequence in token_sequences if sequence]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs: Any) -> bool:
        tokens = input_ids[0].tolist()
        return any(len(tokens) >= len(end) and tokens[-len(end) :] == end for end in self.token_sequences)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows[:limit] if limit is not None else rows


def choose_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return requested


def end_token_sequences(tokenizer: Any) -> list[list[int]]:
    variants = ["</svg>", "</svg>\n", "</svg>\n```"]
    sequences: list[list[int]] = []
    for value in variants:
        encoded = tokenizer(value, add_special_tokens=False)["input_ids"]
        if encoded not in sequences:
            sequences.append(encoded)
    return sequences


def choose_dtype(requested: str, device: str) -> torch.dtype:
    mapping = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    if requested != "auto":
        return mapping[requested]
    if device == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    # Gemma 3 270M produced non-finite logits with FP16 on the local MX450.
    return torch.float32


def load_model(
    model_path: str, adapter_path: str | None, device: str, dtype: torch.dtype
) -> tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        dtype=dtype,
        device_map=device,
    )
    if adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path, local_files_only=True)
    model.eval()
    return model, tokenizer


def generation_config(max_new_tokens: int) -> dict[str, Any]:
    return {
        "do_sample": False,
        "max_new_tokens": max_new_tokens,
        "num_beams": 1,
        "repetition_penalty": 1.1,
        "use_cache": True,
    }


def generate_one(
    model: Any,
    tokenizer: Any,
    messages: list[dict[str, str]],
    max_new_tokens: int,
    device: str,
) -> tuple[str, int, int, float]:
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(rendered, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    stop = StopOnTokenSequence(end_token_sequences(tokenizer))
    started = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            stopping_criteria=StoppingCriteriaList([stop]),
            pad_token_id=tokenizer.eos_token_id,
            **generation_config(max_new_tokens),
        )
    elapsed = time.perf_counter() - started
    output_ids = generated[0, input_ids.shape[1] :]
    output = tokenizer.decode(output_ids, skip_special_tokens=True)
    return output, int(input_ids.shape[1]), int(output_ids.shape[0]), elapsed


def load_completed(path: Path, run_name: str, resume: bool) -> dict[int, dict[str, Any]]:
    if not resume or not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("run_name") != run_name:
        raise ValueError(f"Detailed result belongs to run {payload.get('run_name')!r}, not {run_name!r}")
    return {int(sample["index"]): sample for sample in payload.get("samples", [])}


def save_detailed(path: Path, metadata: dict[str, Any], samples: list[dict[str, Any]]) -> None:
    payload = {**metadata, "samples": samples}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def compact_sample(sample: dict[str, Any]) -> dict[str, Any]:
    reward_result = sample["reward"]
    return {
        "index": sample["index"],
        "prompt_sha256": sample["prompt_sha256"],
        "reward": reward_result["total"],
        "components": {name: reward_result[name] for name in COMPONENTS},
        "valid_xml": reward_result["valid_xml"],
        "drawing_elements": reward_result["drawing_elements"],
        "visible_elements": reward_result["visible_elements"],
        "reasons": reward_result["reasons"],
        "input_tokens": sample["input_tokens"],
        "output_tokens": sample["output_tokens"],
        "generation_seconds": sample["generation_seconds"],
    }


def summarize(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        return {}
    rewards = [float(sample["reward"]["total"]) for sample in samples]
    component_means = {
        name: round(statistics.fmean(float(sample["reward"][name]) for sample in samples), 6)
        for name in COMPONENTS
    }
    return {
        "num_samples": len(samples),
        "proxy_reward_mean": round(statistics.fmean(rewards), 6),
        "proxy_reward_median": round(statistics.median(rewards), 6),
        "valid_xml_rate": round(statistics.fmean(float(sample["reward"]["valid_xml"]) for sample in samples), 6),
        "visible_drawing_rate": round(
            statistics.fmean(float(sample["reward"]["visible_elements"] > 0) for sample in samples), 6
        ),
        "component_means": component_means,
        "output_tokens_mean": round(statistics.fmean(sample["output_tokens"] for sample in samples), 2),
        "generation_seconds_total": round(sum(sample["generation_seconds"] for sample in samples), 2),
    }


def prompt_hash(prompt: str) -> str:
    import hashlib

    return hashlib.sha256(prompt.strip().encode("utf-8")).hexdigest()


def update_compact_results(
    path: Path,
    run_name: str,
    metadata: dict[str, Any],
    samples: list[dict[str, Any]],
) -> None:
    payload = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"schema_version": 1, "runs": {}}
    payload.setdefault("runs", {})[run_name] = {
        "metadata": {key: value for key, value in metadata.items() if key != "run_name"},
        "summary": summarize(samples),
        "samples": [compact_sample(sample) for sample in samples],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/gemma-3-270m-it")
    parser.add_argument("--adapter")
    parser.add_argument("--data", type=Path, default=Path("data/valid.jsonl"))
    parser.add_argument("--run-name", default="base")
    parser.add_argument("--results", type=Path, default=Path("results.json"))
    parser.add_argument("--detailed-results", type=Path, default=Path("results_detailed.json"))
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    set_seed(SEED)
    device = choose_device(args.device)
    dtype = choose_dtype(args.dtype, device)
    rows = load_jsonl(args.data, args.limit)
    model, tokenizer = load_model(args.model, args.adapter, device, dtype)
    generation = generation_config(args.max_new_tokens)
    metadata = {
        "run_name": args.run_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "base_model": args.model,
        "adapter": args.adapter,
        "dataset": args.data.as_posix(),
        "seed": SEED,
        "device": device,
        "dtype": str(next(model.parameters()).dtype),
        "generation": generation,
        "stop_sequence": "</svg>",
        "reward_weights": WEIGHTS,
    }
    completed = load_completed(args.detailed_results, args.run_name, args.resume)
    if completed:
        # Recompute proxy scores so saved generations remain usable after a
        # reward implementation update; generation itself is not repeated.
        for sample in completed.values():
            reward_result = score_svg(sample["prompt"], sample["raw_output"])
            sample["reward"] = reward_result
            sample["extracted_svg"] = reward_result["extracted_svg"]

    for index, row in enumerate(rows, start=1):
        if index in completed:
            print(f"[{index}/{len(rows)}] resumed")
            continue
        messages = row["messages"]
        prompt = messages[1]["content"]
        output, input_tokens, output_tokens, elapsed = generate_one(
            model, tokenizer, messages[:2], args.max_new_tokens, device
        )
        reward_result = score_svg(prompt, output)
        completed[index] = {
            "index": index,
            "prompt": prompt,
            "prompt_sha256": prompt_hash(prompt),
            "reference_svg": messages[2]["content"],
            "raw_output": output,
            "extracted_svg": reward_result["extracted_svg"],
            "reward": reward_result,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "generation_seconds": round(elapsed, 3),
        }
        ordered = [completed[key] for key in sorted(completed)]
        save_detailed(args.detailed_results, metadata, ordered)
        print(
            f"[{index}/{len(rows)}] tokens={output_tokens} seconds={elapsed:.2f} "
            f"reward={reward_result['total']:.3f} valid={reward_result['valid_xml']}"
        )

    ordered = [completed[key] for key in sorted(completed)]
    save_detailed(args.detailed_results, metadata, ordered)
    update_compact_results(args.results, args.run_name, metadata, ordered)
    print(json.dumps(summarize(ordered), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
