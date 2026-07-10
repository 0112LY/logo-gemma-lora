"""Minimal reproducible LoRA trainer using Transformers and PEFT.

This is the fallback backend for environments where the installed ms-swift and
PyTorch versions are incompatible. It consumes the same YAML experiment files
and masks all system/user tokens from the language-model loss.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_rows(paths: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw_path in paths:
        path = Path(raw_path)
        rows.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    return rows


class ChatDataset(Dataset[dict[str, list[int]]]):
    def __init__(self, rows: list[dict[str, Any]], tokenizer: Any, max_length: int, truncate_right: bool = False) -> None:
        self.samples: list[dict[str, list[int]]] = []
        self.deleted = 0
        self.truncated = 0
        for row in rows:
            messages = row["messages"]
            prompt_text = tokenizer.apply_chat_template(messages[:2], tokenize=False, add_generation_prompt=True)
            full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
            input_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
            if len(input_ids) > max_length:
                if truncate_right:
                    input_ids = input_ids[:max_length]
                    self.truncated += 1
                else:
                    self.deleted += 1
                    continue
            labels = [-100] * min(len(prompt_ids), len(input_ids)) + input_ids[len(prompt_ids) :]
            if not any(label != -100 for label in labels):
                self.deleted += 1
                continue
            self.samples.append({"input_ids": input_ids, "labels": labels})
        if not self.samples:
            raise ValueError(f"no samples fit max_length={max_length}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.samples[index]


class Collator:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, samples: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        width = max(len(sample["input_ids"]) for sample in samples)
        input_ids: list[list[int]] = []
        labels: list[list[int]] = []
        attention_mask: list[list[int]] = []
        for sample in samples:
            padding = width - len(sample["input_ids"])
            input_ids.append(sample["input_ids"] + [self.pad_token_id] * padding)
            labels.append(sample["labels"] + [-100] * padding)
            attention_mask.append([1] * len(sample["input_ids"]) + [0] * padding)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


def evaluate(model: Any, loader: DataLoader[Any], device: str, dtype: torch.dtype) -> float:
    model.eval()
    losses: list[float] = []
    context = torch.autocast(device_type="cuda", dtype=dtype) if device == "cuda" else nullcontext()
    with torch.inference_mode(), context:
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            loss = model(**batch).loss
            losses.append(float(loss.detach().float().cpu()))
    model.train()
    return sum(losses) / len(losses)


def save_adapter(model: Any, tokenizer: Any, path: Path, state: dict[str, Any]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path, safe_serialization=True)
    tokenizer.save_pretrained(path)
    (path / "trainer_state.json").write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--gradient-accumulation-steps", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument(
        "--truncate-right",
        action="store_true",
        help="Smoke-test only: truncate long rows instead of applying the formal delete strategy.",
    )
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    for name, value in config.get("ENV", {}).items():
        os.environ.setdefault(str(name), str(value))
    seed = int(config.get("seed", 42))
    set_seed(seed)
    if not torch.cuda.is_available():
        raise RuntimeError("LoRA training requires CUDA in this project")
    device = "cuda"
    dtype_name = config.get("torch_dtype", "bfloat16")
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype_name]
    max_length = args.max_length or int(config["max_length"])
    grad_accum = args.gradient_accumulation_steps or int(config["gradient_accumulation_steps"])
    output_dir = args.output_dir or Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(config["model"], local_files_only=True)
    train_dataset = ChatDataset(load_rows(config["dataset"]), tokenizer, max_length, args.truncate_right)
    val_dataset = ChatDataset(load_rows(config["val_dataset"]), tokenizer, max_length, args.truncate_right)
    collator = Collator(tokenizer.pad_token_id)
    generator = torch.Generator().manual_seed(int(config.get("data_seed", seed)))
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["per_device_train_batch_size"]),
        shuffle=True,
        generator=generator,
        num_workers=int(config.get("dataloader_num_workers", 0)),
        collate_fn=collator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(config["per_device_eval_batch_size"]),
        shuffle=False,
        num_workers=0,
        collate_fn=collator,
    )

    model = AutoModelForCausalLM.from_pretrained(
        config["model"], local_files_only=True, dtype=dtype, attn_implementation=config.get("attn_impl", "sdpa")
    )
    model.config.use_cache = False
    if config.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    lora = LoraConfig(
        task_type="CAUSAL_LM",
        r=int(config["lora_rank"]),
        lora_alpha=int(config["lora_alpha"]),
        lora_dropout=float(config["lora_dropout"]),
        target_modules=TARGET_MODULES,
        bias="none",
    )
    model = get_peft_model(model, lora).to(device)
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    print(f"train rows={len(train_dataset)} deleted={train_dataset.deleted} truncated={train_dataset.truncated}")
    print(f"valid rows={len(val_dataset)} deleted={val_dataset.deleted} truncated={val_dataset.truncated}")
    print(f"parameters trainable={trainable} total={total} percent={100 * trainable / total:.4f}")

    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=float(config["learning_rate"]),
        weight_decay=float(config.get("weight_decay", 0.1)),
        betas=(0.9, 0.95),
    )
    epochs = int(config["num_train_epochs"])
    natural_steps = math.ceil(len(train_loader) / grad_accum) * epochs
    total_steps = args.max_steps or natural_steps
    warmup_steps = max(1, round(total_steps * float(config.get("warmup_ratio", 0.05))))
    scheduler = get_scheduler(
        config.get("lr_scheduler_type", "cosine"),
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    eval_steps = int(config.get("eval_steps", 5))
    save_steps = int(config.get("save_steps", 5))
    log_path = output_dir / "train_log.jsonl"
    global_step = 0
    micro_step = 0
    accumulated_loss = 0.0
    accumulated_batches = 0
    best_eval_loss = float("inf")
    best_checkpoint: str | None = None
    started = time.perf_counter()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    context = torch.autocast(device_type="cuda", dtype=dtype)

    for epoch in range(epochs):
        for batch_index, batch in enumerate(train_loader, start=1):
            batch = {key: value.to(device) for key, value in batch.items()}
            with context:
                raw_loss = model(**batch).loss
                loss = raw_loss / grad_accum
            if not torch.isfinite(raw_loss):
                raise FloatingPointError(f"non-finite training loss: {raw_loss}")
            loss.backward()
            micro_step += 1
            accumulated_loss += float(raw_loss.detach().float().cpu())
            accumulated_batches += 1
            end_of_epoch = batch_index == len(train_loader)
            if micro_step % grad_accum and not end_of_epoch:
                continue
            if end_of_epoch and accumulated_batches < grad_accum:
                correction = grad_accum / accumulated_batches
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.mul_(correction)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            record: dict[str, Any] = {
                "step": global_step,
                "epoch": epoch + 1,
                "train_loss": round(accumulated_loss / accumulated_batches, 6),
                "learning_rate": scheduler.get_last_lr()[0],
                "elapsed_seconds": round(time.perf_counter() - started, 3),
                "cuda_memory_mib": round(torch.cuda.max_memory_allocated() / 2**20, 2),
            }
            if not args.skip_eval and global_step % eval_steps == 0:
                record["eval_loss"] = round(evaluate(model, val_loader, device, dtype), 6)
                if record["eval_loss"] < best_eval_loss:
                    best_eval_loss = record["eval_loss"]
                    best_checkpoint = str(output_dir / f"checkpoint-{global_step}")
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            print(json.dumps(record))
            accumulated_loss = 0.0
            accumulated_batches = 0
            if global_step % save_steps == 0:
                save_adapter(model, tokenizer, output_dir / f"checkpoint-{global_step}", record)
            if global_step >= total_steps:
                break
        if global_step >= total_steps:
            break

    final_state = {
        "global_step": global_step,
        "best_eval_loss": None if math.isinf(best_eval_loss) else best_eval_loss,
        "best_checkpoint": best_checkpoint,
        "max_length": max_length,
        "seed": seed,
        "config": config,
    }
    save_adapter(model, tokenizer, output_dir / "final", final_state)
    print(json.dumps(final_state, indent=2))


if __name__ == "__main__":
    main()
