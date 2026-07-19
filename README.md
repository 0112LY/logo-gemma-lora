# Gemma 3 270M Logo SVG LoRA

本仓库是自然语言生成 SVG Logo 的 LoRA 微调作业。基座模型为 `google/gemma-3-270m-it`，训练框架为 ms-swift。仓库提供最终 adapter、SVG 代理奖励、训练配置、验证结果和实验报告。

## 文件说明

| 路径 | 内容 |
| --- | --- |
| `.gitignore` | 本地数据、模型和训练中间产物的忽略规则 |
| `adapter/adapter_config.json` | 最终 LoRA adapter 配置 |
| `adapter/adapter_model.safetensors` | 最终 LoRA 权重 |
| `reward.py` | 作业要求的 reward 入口 |
| `student_kit/__init__.py` | `student_kit` 包入口 |
| `student_kit/reward.py` | SVG reward 的完整实现 |
| `student_kit/eval_self.py` | 验证集生成与评分脚本 |
| `student_kit/train_peft.py` | PEFT 训练辅助脚本 |
| `train_config.yaml` | 最终 ms-swift 训练配置 |
| `configs/exp1_rank8_lr1e-4.yaml` | rank 8、学习率 1e-4 的 baseline 配置 |
| `configs/exp2_rank16_lr1e-4.yaml` | rank 16 对比配置 |
| `configs/exp3_rank8_lr2e-4.yaml` | 学习率 2e-4 对比配置 |
| `configs/exp4_final_lr5e-6.yaml` | 生成退化后追加的保守最终配置 |
| `results.json` | 最终验证集汇总和逐样本评分 |
| `report.md` | 实验过程、微调前后对比、失败分析与结论 |
| `requirements.txt` | Python 依赖版本 |
| `scripts/inspect_data.py` | 数据格式与统计检查 |
| `scripts/prepare_data.py` | 训练数据预处理 |
| `tests/test_reward.py` | reward 单元测试 |

## 运行方式

```bash
pip install -r requirements.txt
python scripts/prepare_data.py
swift sft train_config.yaml
python -m unittest discover -s tests -v
```

完整实验说明见 [report.md](report.md)。
