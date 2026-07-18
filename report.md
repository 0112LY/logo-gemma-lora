# Gemma 3 270M Logo SVG LoRA 实验报告

## 1. 任务与目标

实验任务：使用 Gemma 3 270M 指令模型，根据自然语言提示生成可解析、可见且尽量符合提示的 SVG Logo。提交内容包括 LoRA adapter、代理奖励实现、最终训练配置、验证结果和实验报告。

实验目标：用可解释、可测试的结构指标判断输出是否具备基本可用性，并分析 LoRA 相对基座的行为。

## 2. 数据与模型

- 基座模型：ModelScope `google/gemma-3-270m-it`
- 本地模型目录：`models/gemma-3-270m-it`
- 训练数据：上游 `train.jsonl`，过滤两条 prompt 为 `placeholder` 的记录后共 217 条
- 验证数据：`valid.jsonl`，共 17 条
- 最长序列：3584 token；超长记录使用 `delete` 策略
- 最终训练硬件：NVIDIA A10，约 22.18 GiB 可用显存
- 最终训练精度：BF16


## 3. Reward 设计

reward 的权重为：有效性 30%、结构 15%、画布 15%、可见性 15%、简洁性 10%、提示词保真 10%、安全性 5%。主要检查包括：

1. 提取完整 SVG，检查 XML 可解析且根节点为 `svg`；
2. 检查 `viewBox` 或正尺寸的 `width/height`；
3. 要求存在基础绘图元素及可见 fill/stroke；
4. 检查基础几何数值、path 命令、异常坐标和元素数量；
5. 拒绝脚本、危险标签、事件属性和外部链接；
6. 检测重复属性、重复片段和异常长输出；
7. 对提示中的颜色、基础形状和文本要求做轻量匹配。

该 reward 不计算曲线精确边界，不进行真实渲染，也不能充分衡量构图、审美和语义。因此不把分数提升解释为视觉质量提升。

## 4. 实验过程

### 4.1 BF16 基座

BF16 基座能够自然结束并生成完整 SVG，但质量不稳定。单样本诊断中，模型生成了完整的 `<svg>...</svg>`，同时也出现画布外矩形、纯白图形或无可见元素。

早期本地评估曾记录平均 reward 0.065882、XML 有效率 11.76%、可见图形率 0%。该结果使用 `student_kit/eval_self.py`、2048 token 上限和另一条推理路径；它与最终 ms-swift/合并模型、1024 token 上限的结果口径不同，不能作为最终提升量计算。

### 4.2 初始 LoRA

| 实验 | 主要配置 | 训练记录 | 自由生成现象 |
| --- | --- | --- | --- |
| Exp1 | BF16，rank 8，alpha 16，all-linear，lr 1e-4 | train loss 1.177，最终 eval loss 1.055 | 17 条均达到 2048 token；重复 SVG 属性，无完整闭合 |
| Exp4 | BF16，rank 8，alpha 16，all-linear，lr 5e-5 | train loss 1.29，最终 eval loss 1.174 | checkpoint 5/10/15 分别重复 0、10 或 XML namespace |

训练 loss 和 eval loss 均下降，但自由生成反而崩溃。这说明 teacher-forcing 的 token loss 与 SVG 可用性并不一致，不能仅依据最低 eval loss 选择 checkpoint。

### 4.3 Q4/QLoRA 诊断

使用 bitsandbytes 0.49.2 尝试 NF4 4-bit：

- Exp5：rank 8、`q_proj/v_proj`、lr 2e-5，train loss 1.895，最佳 eval loss 1.921；
- Exp6：rank 4、alpha 4、`q_proj/v_proj`、lr 5e-6，仅训练 5 步。

两组 adapter 推理都出现 256、16 等数字重复。随后对“不加载 adapter 的 Q4 基座”做对照，Q4 基座本身同样重复到 1024 token 且无法闭合 SVG。因此该问题不能归因于 LoRA；本实验将其判断为当前 Gemma 3 270M、Transformers 5.8.1、ms-swift 4.4.0 与 bitsandbytes 0.49.2 组合下的量化兼容性或量化质量问题，并回退到 BF16。

### 4.4 BF16 超保守诊断与 checkpoint 选择

Exp7 使用 rank 4、alpha 4、`q_proj/v_proj`、lr 5e-6、BF16，每步保存，仅训练 5 个优化步骤。训练过程数值稳定：loss 约 1.62–1.91，grad norm 约 1.26–1.41，无 NaN/Inf。

关键诊断结果如下：

- checkpoint-1 的 36 个 `lora_B` 张量全部为精确零，因此 LoRA 增量 `B×A` 为零；
- checkpoint-1 运行时挂载 adapter 会异常重复，但合并后恢复为正常基座输出，表明当前运行时 adapter 路径还存在兼容性问题；
- checkpoint-5 的 36 个 `lora_B` 均已非零，最大绝对值约 `1.25e-5`；合并后仍重复数字 0，`repetition_penalty=1.1` 也未修复。

因此所有观测到的非零 LoRA 更新都降低了生成稳定性。最终选择 checkpoint-1 作为最稳定提交，但必须明确：它是合法 adapter 文件，却是零增量 adapter，表现等同 BF16 基座，不能视为微调提升。

## 5. 最终验证结果

最终验证使用合并后的 checkpoint-1、BF16、贪心解码、temperature 0、repetition penalty 1.0、最大新 token 1024。17 条结果记录在 `results.json`。

| 指标 | 数值 |
| --- | ---: |
| 样本数 | 17 |
| 平均代理 reward | 0.290784 |
| reward 中位数 | 0.200000 |
| XML 有效率 | 52.94%（9/17） |
| SVG 闭合率 | 47.06%（8/17） |
| 可见图形率 | 23.53%（4/17） |

4 条样本生成了可见绘图元素，其中最高 reward 为 0.966667；其余失败主要来自缺少完整 SVG、XML 非法、没有绘图元素或没有显式可见颜色。最终 checkpoint 与同口径 BF16 基座理论等价，所以本实验的可靠结论是“未观察到 LoRA 提升”，而不是从早期异口径 Base 数字推导出的表面增幅。

## 6. 失败分析与局限

1. **小模型生成敏感。** 270M 模型在长结构化代码生成中容易进入数字或属性循环，微小参数更新也可能改变贪心解码轨迹。
2. **训练 loss 不能代替生成评估。** Exp1 的 eval loss 持续下降，但所有验证输出都达到长度上限。
3. **Q4 对照失败。** 未微调 Q4 基座已经退化，因此本次 QLoRA 结果不具备解释训练效果的条件。
4. **运行时 adapter 兼容问题。** 零增量 adapter 在直接挂载时改变输出，而合并后恢复正常；这说明当前软件组合的推理路径存在额外风险。
5. **代理 reward 有限。** XML 合法、元素可见并不等于图形美观或语义正确；提示词匹配只是弱代理，也可能漏判同义表达。
6. **实验规模有限。** 只有 217 条训练记录和 17 条验证记录，没有进行人工盲评、置信区间或完整 reward 消融。

## 7. 可复现性

最终核心环境和硬件记录在 `ENVIRONMENT.md`，依赖记录在 `requirements.txt`，最终训练参数位于 `train_config.yaml`。主要命令为：

```bash
python scripts/prepare_data.py
swift sft train_config.yaml
python -m pytest -q
```

提交的 `adapter/` 包含 PEFT 配置与 safetensors 权重；模型本体、原始数据、缓存、中间 checkpoint 和临时合并模型未提交。

## 8. 总结

本项目完成了数据审计、可解释 SVG reward、BF16 LoRA、Q4 对照、checkpoint 级生成诊断和最终验证。实验没有取得可靠的 LoRA 提升，但通过基座对照、Q4 基座对照、零/非零 LoRA 权重检查和合并推理，定位了生成退化与训练 loss 脱节、量化路径异常和 adapter 运行时兼容问题。

