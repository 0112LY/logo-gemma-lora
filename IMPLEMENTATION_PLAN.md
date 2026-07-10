# Implementation plan

This plan intentionally targets a small, reproducible course project. Features
that do not materially improve the required comparison are deferred.

## Final repository structure

```text
logo-gemma-lora/
├── README.md
├── requirements.txt
├── ENVIRONMENT.md
├── reward.py
├── train_config.yaml
├── results.json
├── report.md
├── adapter/
│   ├── adapter_config.json
│   └── adapter_model.safetensors
├── student_kit/
│   ├── reward.py
│   ├── eval_self.py
│   └── ...
├── scripts/
│   ├── inspect_data.py
│   ├── inspect_results.py
│   └── render_examples.py
├── examples/
│   └── selected SVG comparisons only
└── tests/
    └── test_reward.py
```

The following stay local and are ignored by Git: the base model, raw datasets,
upstream checkout, virtual environment, intermediate checkpoints, bulk
generations, full experiment outputs, caches, and figures. Only the final LoRA
adapter and a few report examples are submitted.

## Stage 1: environment and data

Status: complete.

- Use ModelScope model `google/gemma-3-270m-it` from
  `models/gemma-3-270m-it`.
- Pin the verified software stack in `requirements.txt` and record the hardware
  in `ENVIRONMENT.md`.
- Use random seed 42 throughout training and evaluation.
- Audit the source data with `scripts/inspect_data.py`.
- Keep source JSONL unchanged. Exclude the two low-information training rows
  whose prompt is `placeholder` when producing the training input.
- Prefer `max_length=3584` on AI Studio to retain all published samples. Use
  2048 only for a local smoke test when memory is constrained.

## Stage 2: reward v1

The first reward implementation is deliberately limited to robust checks that
can be explained and tested without building an SVG renderer.

### Required checks

1. Extract one complete SVG from plain output or a Markdown code block.
2. Parse XML safely and require an `svg` root.
3. Require at least one visible drawing element.
4. Validate `viewBox`, or positive `width` and `height` when no viewBox exists.
5. Recognize visible colors in `fill`, `stroke`, and inline `style`.
6. Check that the number of drawing elements is non-zero and not excessive.
7. Reject dangerous tags, event handlers, scripts, and external links.
8. Detect obvious repeated-output degeneration and extreme output length.
9. Validate basic shapes with simple numeric attributes.
10. Apply lightweight prompt matching for explicit colors, basic shapes, and
    requested text.

For `path`, v1 only checks that `d` is non-empty, contains recognized SVG path
commands, has finite numeric values, and does not contain extreme coordinates.
It does not compute Bezier or arc geometry.

### Initial weights

| Component | Weight |
| --- | ---: |
| XML/SVG validity | 30% |
| Tags and structure | 15% |
| Canvas and coordinates | 15% |
| Visibility and colors | 15% |
| Element count and simplicity | 10% |
| Verifiable prompt fidelity | 10% |
| Degeneration and safety | 5% |

Hard gates cap or zero the result for missing SVG, unparseable XML, dangerous
content, or no visible drawing. Prompt fidelity is only a weak proxy and must
not inspect `id`, `class`, `title`, or comments for keyword credit.

### Deferred enhancements

- Complete `defs`, gradient, and clip-path reference validation.
- Geometric path bounding boxes or rendering-based boundary checks.
- Foreground/background contrast estimation.
- Dataset-wide generated-SVG diversity metrics.
- Bootstrap confidence intervals.
- Reward ablation experiments.

These are added only if the required pipeline is complete and an observed
failure justifies the extra implementation.

## Stage 3: baseline evaluation

Status: complete.

- Evaluate the unmodified base model on the 17 validation prompts.
- Fix the chat template, seed, generation length, temperature, and sampling
  settings for every comparison.
- Store full generations locally in `results_detailed.json`.
- Put only metadata, summary metrics, per-sample component scores, failure
  reasons, validity, and lengths in the submitted `results.json`.

Recorded Base result with seed 42, greedy decoding, and a 2048-token output
limit: proxy reward 0.065882, XML validity 11.76%, and visible drawing rate 0%.
The full run took 695.06 generation seconds locally. Fourteen outputs used a
self-closing SVG form, but most contained malformed attributes; two were valid
empty SVG elements and none contained a visible drawing. This weak baseline is
expected and provides the required comparison point for LoRA.

## Stage 4: compact LoRA experiment set

Status: configured and smoke-tested; full-length runs require AI Studio.

Run only these planned comparisons initially:

| Run | Rank | Learning rate | Epochs | Purpose |
| --- | ---: | ---: | ---: | --- |
| Base | - | - | - | Required untrained reference |
| Exp1 | 8 | 1e-4 | 1 | Minimal LoRA baseline |
| Exp2 | 16 | 1e-4 | 1 | Rank comparison |
| Exp3 | 8 | 2e-4 | 1 | Learning-rate comparison |

Use assistant-only loss masking and the same filtered training set. Compare
validation loss, total proxy reward, validity, and prompt-fidelity component.
Select the most stable checkpoint rather than assuming the last checkpoint is
best. Add another run only when one of these results shows a specific problem,
such as immediate overfitting or generation collapse.

The preferred backend is ms-swift 4.4.0. It successfully loads the model,
tokenizes assistant-only labels, and injects LoRA. The local PyTorch 2.5.1
build cannot create its Trainer because that version lacks
`torch.distributed.fsdp.FSDPModule`, so `student_kit/train_peft.py` remains a
tested fallback for the local machine. The AI Studio PyTorch 2.10 environment
provides this API and should run the YAML files directly with `swift sft`.

All three configurations completed a one-step local smoke test at 320 tokens.
This smoke-only truncation is not an experiment result. At 512 and 1024 tokens,
the MX450 ran out of memory because the 262k-token vocabulary makes the
training loss materialize large FP32 logits. The formal 3584-token runs must be
executed on an AI Studio GPU with:

```bash
python scripts/prepare_data.py
swift sft configs/exp1_rank8_lr1e-4.yaml
swift sft configs/exp2_rank16_lr1e-4.yaml
swift sft configs/exp3_rank8_lr2e-4.yaml
```

## Stage 5: final evaluation and submission

- Re-run Base and Final using identical decoding settings.
- Report proxy-reward deltas and component-level changes.
- Manually inspect a small set of improved, unchanged, and degraded examples.
- Keep `results.json` compact; place full text/SVG outputs in the ignored
  `results_detailed.json` and commit only selected examples.
- Save only the final `adapter_config.json` and `adapter_model.safetensors`.
- Verify adapter loading and reproduce the submitted summary from a clean run.

## Reporting language

The report distinguishes the local proxy from the hidden evaluation. Preferred
wording is:

> On the local proxy reward, LoRA improved SVG syntax validity and the visible
> drawing rate. Prompt semantic fidelity remained limited, so the proxy gain
> does not necessarily imply an improvement under the hidden visual review.

Claims of improved visual quality require evidence from manual inspection and
must not be inferred from the proxy score alone.

## Commit milestones

1. Initial rollback point. (Complete locally.)
2. Reward v1, tests, environment records, and data-inspection tooling.
3. Final `train_config.yaml`.
4. Final adapter and compact `results.json`.
5. `report.md` and `README.md`.

Push each milestone when network access to GitHub is available, followed by a
final verification push.
