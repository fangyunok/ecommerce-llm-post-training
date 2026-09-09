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

- `data/processed/sft_train.jsonl`：1800条；
- `data/processed/sft_validation.jsonl`：225条；
- 九类任务均衡分布。

当前数据是可复现的合成启动数据，用于验证训练闭环。后续实验需要人工抽检，并补充更自然、多样、接近真实业务分布的数据。

实验002使用`data/eval/ecommerce_eval_v2.jsonl`的40条人工评测题。该文件保留原8题，并增加预算边界、否定属性、数值筛选和完整摘要题；不得将这些题目或答案复制到训练集。

实验003数据单独生成，不覆盖v2：

```bash
python scripts/generate_sft_data.py --profile v3
python scripts/preflight_training.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --train-file data/processed/sft_train_v3.jsonl \
  --validation-file data/processed/sft_validation_v3.jsonl
python -m src.training.train_qlora \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --train-file data/processed/sft_train_v3.jsonl \
  --validation-file data/processed/sft_validation_v3.jsonl \
  --output-dir outputs/sft_adapter_v3 \
  --epochs 3 --learning-rate 2e-4 --batch-size 2 \
  --gradient-accumulation 8 --max-length 1024 \
  --lora-r 16 --lora-alpha 32
```

## GPU环境

建议使用Linux、Python 3.11和单张NVIDIA GPU。新建独立虚拟环境，避免训练依赖影响已有应用：

```bash
python -m venv .venv-train
source .venv-train/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-training.txt
```

AutoDL已验证配置：单张RTX 3090 24GB，PyTorch 2.5.1、Python 3.12、CUDA 12.4。为复用镜像自带的GPU版PyTorch，可创建继承系统包的环境，并只补齐项目依赖：

```bash
python -m venv --system-site-packages .venv-train
source .venv-train/bin/activate
python -m pip install --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
  -r requirements-training.txt
```

模型与依赖缓存应放到数据盘，避免占满系统盘：

```bash
export HF_HOME=/root/autodl-tmp/huggingface
export HF_ENDPOINT=https://hf-mirror.com
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

## 首轮实验结果

- GPU：RTX 3090 24GB；
- 训练：3 epochs、225 optimizer steps、约11分13秒；
- 峰值显存：约2GB；
- 最终验证loss：0.000131；
- 固定业务评测：基座2/8（25%），SFT后4/8（50%）。

验证loss极低主要反映合成数据模板规律强，不代表真实业务准确率。下一轮优先增加语言表达、属性顺序、否定表达和数值字段的多样性，再保持训练参数不变进行数据对照实验。

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
