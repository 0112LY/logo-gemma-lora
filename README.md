# Logo SVG 生成：Gemma 3 270M LoRA 实验

本项目使用 `google/gemma-3-270m-it` 完成自然语言到 SVG Logo 的生成实验。项目包含可解释的 SVG 代理奖励、可复现训练配置、LoRA adapter、验证集评分结果和实验报告。

## 最终结论

实验没有观察到可靠的 LoRA 提升。BF16 基座可以在部分提示上生成完整 SVG，但所有观测到的非零 LoRA checkpoint 都出现了数字、属性或命名空间重复，无法稳定闭合 SVG。最终提交选择最稳定的 `checkpoint-1`；检查表明其 36 个 `lora_B` 张量均为零，因此它与 BF16 基座等价，不能宣称为微调提升。

最终 17 条验证样本的本地代理指标为：

| 指标 | 结果 |
| --- | ---: |
| 平均代理 reward | 0.290784 |
| reward 中位数 | 0.200000 |
| XML 有效率 | 52.94%（9/17） |
| SVG 闭合率 | 47.06%（8/17） |
| 可见图形率 | 23.53%（4/17） |

这些指标只衡量可程序验证的 SVG 结构，不等价于真实视觉质量或隐藏评测成绩。完整分析见 [report.md](report.md)。

## 提交结构

```text
.
├─ adapter/
│  ├─ adapter_config.json
│  └─ adapter_model.safetensors
├─ student_kit/
│  ├─ reward.py
│  ├─ eval_self.py
│  └─ train_peft.py
├─ configs/                 # 历次正式/诊断实验配置
├─ tests/test_reward.py
├─ reward.py                # 提交入口
├─ train_config.yaml        # 最终选择配置
├─ results.json             # 17 条验证样本的紧凑结果
├─ ENVIRONMENT.md
├─ requirements.txt
└─ report.md
```

模型、数据、缓存、合并模型和中间 checkpoint 不提交到 Git。`student_kit` 未包含在上游数据仓库是正常情况，本项目中版本为自行整理的训练与评估工具。

## 环境与数据准备

最终实验在 ModelScope DSW 的 NVIDIA A10 上运行，核心环境为 Python 3.12.13、PyTorch 2.10.0+cu128、ms-swift 4.4.0、Transformers 5.8.1 和 PEFT 0.19.1。详细信息见 [ENVIRONMENT.md](ENVIRONMENT.md)。

安装时应先选择与机器 CUDA 匹配的 PyTorch，再安装其余依赖：

```bash
pip install -r requirements.txt
```

下载模型：

```bash
modelscope download \
  --model google/gemma-3-270m-it \
  --local_dir models/gemma-3-270m-it
```

将上游 `train.jsonl`、`valid.jsonl` 放入 `data/` 后准备训练数据：

```bash
python scripts/inspect_data.py
python scripts/prepare_data.py
```

准备脚本会过滤两条 prompt 为 `placeholder` 的低信息训练记录；`excluded=2` 即来源于此，不是数据读取错误。

## Reward

`reward.py` 对生成结果执行以下检查：

- 提取完整 `<svg>...</svg>` 并解析 XML；
- 检查根节点、画布、可见图形和颜色；
- 检查危险标签、外部链接和事件处理器；
- 对基础图形数值和 path 命令执行轻量合法性检查；
- 检测明显重复退化；
- 对提示词中的颜色、形状和文本做弱语义匹配。

快速调用：

```python
from reward import reward

score = reward("设计一个蓝色圆形标志", "<svg>...</svg>")
```

运行测试：

```bash
python -m pytest -q
```

## 训练与评估

最终配置为 BF16、LoRA rank 4、alpha 4、`q_proj/v_proj`、学习率 `5e-6`、随机种子 42：

```bash
swift sft train_config.yaml
```

当前软件组合中，使用 `swift infer --adapters ...` 运行时挂载 adapter 会产生与权重不一致的重复输出。实验诊断因此先通过 PEFT 的 `merge_and_unload()` 合并到临时 BF16 基座，再进行推理；合并模型只用于诊断，不属于提交物。

项目评估脚本支持可配置重复惩罚：

```bash
python student_kit/eval_self.py \
  --model models/gemma-3-270m-it \
  --adapter adapter \
  --data data/valid.jsonl \
  --max-new-tokens 1024 \
  --repetition-penalty 1.0
```

如需复核提交结果，以 `results.json` 的元数据、逐样本失败原因和 [report.md](report.md) 中的口径说明为准。

