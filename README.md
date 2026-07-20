# Gemma 3 270M SVG 徽标 LoRA

本仓库是使用 Gemma 3 270M 和 ms-swift 完成的 SVG 徽标生成 LoRA 微调作业，包含reward、最终训练配置、适配器、固定验证集评测结果及实验报告。

## 文件说明

| 路径 | 内容 |
| --- | --- |
| `adapter/adapter_config.json` | LoRA 配置 |
| `adapter/adapter_model.safetensors` | LoRA 权重 |
| `reward.py` | reward 入口 |
| `student_kit/reward.py` | SVG 验证与 reward 完整实现 |
| `student_kit/eval_self.py` | 固定解码的验证集生成与评分脚本 |
| `train_config.yaml` | ms-swift 配置 |
| `results.json` | 基座与最终微调模型的验证集指标及 checkpoint 对比 |
| `report.md` | Reward 设计、实验过程、loss、结果和局限分析 |
| `requirements.txt` | 主要依赖版本 |
| `scripts/prepare_data.py` | 训练数据预处理脚本 |
| `tests/test_reward.py` | Reward 单元测试 |

## 运行

```bash
pip install -r requirements.txt
python scripts/prepare_data.py
swift sft train_config.yaml
python -m unittest discover -s tests -v
```

实验结论和固定验证结果见 [report.md](report.md)。
