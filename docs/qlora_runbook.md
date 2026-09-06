# QLoRA训练运行手册

## 当前实验基线

- 固定评测集：8条，覆盖硬约束、属性匹配、拒绝回答和事实摘要。
- Qwen2.5-0.5B-Instruct基座：严格通过2条，准确率25%。
- 基线结果：`outputs/evaluation/baseline.jsonl`。

## 数据

执行：

```bash
python scripts/generate_sft_data.py
python scripts/preflight_training.py --model /path/to/Qwen2.5-0.5B-Instruct
```

默认得到：

- `data/processed/sft_train.jsonl`：1200条；
- `data/processed/sft_validation.jsonl`：150条；
- 五类任务均衡分布。

当前数据是可复现的合成启动数据，用于验证训练闭环。正式求职版本需要人工抽检，并补充更自然、多样、接近真实业务分布的数据。

## GPU环境

建议使用Linux、Python 3.11和单张NVIDIA GPU。新建独立虚拟环境，避免训练依赖影响已有应用：

```bash
python -m venv .venv-train
source .venv-train/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-training.txt
```

先执行体检，再训练：

```bash
python scripts/preflight_training.py --model Qwen/Qwen2.5-0.5B-Instruct
python -m src.training.train_qlora \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --epochs 3 \
  --learning-rate 2e-4 \
  --batch-size 2 \
  --gradient-accumulation 8 \
  --max-length 1024 \
  --lora-r 16 \
  --lora-alpha 32
```

有效批大小为：`per_device_batch_size × gradient_accumulation × GPU数量`。当前单卡配置为`2 × 8 × 1 = 16`。

## 首轮参数及调试顺序

1. 先保持`r=16, alpha=32, dropout=0.05`不变，验证loss正常下降且无OOM。
2. 若OOM，依次降低`batch-size`、降低`max-length`、提高梯度累积；不要先随意删LoRA目标层。
3. 若训练loss下降而验证loss持续上升，减少epoch或学习率，并增加数据多样性。
4. 若训练和验证loss都几乎不降，检查数据格式、completion loss掩码，并尝试将学习率从`2e-4`调到`1e-4`或`3e-4`做单变量对比。
5. 选模以固定业务评测集为主，`eval_loss`只作为辅助，不把最低loss自动等同于最好业务效果。

训练输出是LoRA Adapter，默认保存至`outputs/sft_adapter`。

## 微调后部署与对比

将Adapter复制回部署机器，启动：

```powershell
$env:ADAPTER_PATH = "D:\面试项目\outputs\sft_adapter"
powershell -ExecutionPolicy Bypass -File scripts\start_api.ps1
```

再次执行：

```powershell
python -m src.evaluation.evaluate_api `
  --output outputs/evaluation/sft.jsonl
```

必须同时保留`baseline.jsonl`和`sft.jsonl`，比较严格准确率、各类别错误、输出长度和延迟。

