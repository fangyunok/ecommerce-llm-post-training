# 电商大模型后训练与推理服务

这是一个面向算法实习求职的可复现实验项目。目标是完成：

`业务定义 → 数据构造 → 基座评测 → QLoRA/SFT → DPO → 自动评测 → 推理服务`

## 项目状态

| 模块 | 状态 | 结果 |
|---|---|---|
| 本地CPU推理API | 已完成 | FastAPI健康检查与聊天接口可用 |
| 固定业务评测 | 已完成 | 基座严格准确率2/8（25%） |
| 合成SFT数据 | 已完成 | 1200条训练、150条验证 |
| QLoRA训练代码 | 已完成，待GPU执行 | 本机无CUDA，尚无Adapter和微调后指标 |
| DPO偏好对齐 | 未开始 | 后续阶段 |
| RAG与购物Agent | 未开始 | 后续阶段 |

> 当前仓库不能声称已经完成模型微调；已完成的是可运行的基座部署、评测闭环和GPU训练入口。

## 当前可运行服务

本地CPU基线使用 `Qwen/Qwen2.5-0.5B-Instruct + Transformers + FastAPI`。

### 从GitHub克隆后安装

```powershell
git clone <你的仓库地址>
cd <仓库目录>
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-serving.txt
```

模型权重不会提交到GitHub。若本地没有`models/Qwen2.5-0.5B-Instruct`，服务首次启动会从Hugging Face下载；也可以在国内网络使用ModelScope提前下载：

```powershell
python -m pip install modelscope
modelscope download --model Qwen/Qwen2.5-0.5B-Instruct --local_dir models\Qwen2.5-0.5B-Instruct
```

### 启动服务

```powershell
cd <项目目录>
powershell -ExecutionPolicy Bypass -File scripts\start_api.ps1
```

启动后打开 `http://127.0.0.1:8000/docs`，或在另一个PowerShell执行：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\smoke_test.ps1
```

主要接口：

- `GET /health`：模型状态、设备与加载错误。
- `POST /v1/chat/completions`：聊天生成，返回token用量和耗时。

若后续已有LoRA Adapter，可设置 `ADAPTER_PATH` 后复用同一服务。

### 运行固定基线评测

保持API运行，在另一个PowerShell执行：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\evaluate_baseline.ps1
```

评测结果保存到 `outputs/evaluation/baseline.jsonl`，后续微调模型使用同一评测集对比。

当前基座评测结果：`2/8`严格通过，准确率`25%`。

### 生成SFT数据

```powershell
python scripts\generate_sft_data.py
```

默认生成1200条训练数据和150条验证数据，格式为对话式prompt/completion。

### 在NVIDIA GPU环境进行QLoRA

```powershell
python -m pip install -r requirements-training.txt
powershell -ExecutionPolicy Bypass -File scripts\train_qlora.ps1
```

本机没有CUDA，训练命令应在Colab、AutoDL或实验室GPU服务器执行。训练完成后将`outputs/sft_adapter`作为`ADAPTER_PATH`接入现有API。

训练前先阅读并执行[QLoRA运行手册](docs/qlora_runbook.md)中的环境体检与参数检查。

## 后续实验路线

1. 在NVIDIA GPU环境完成首轮QLoRA/SFT训练，保存Adapter、训练日志和超参数。
2. 使用固定评测集对比基座模型与SFT模型，报告严格准确率、延迟和失败案例。
3. 根据错误分析清洗或补充训练数据，进行至少一轮可解释的参数对照实验。
4. 构造chosen/rejected偏好数据，增加DPO训练并与SFT结果对比。
5. 最后根据业务需要增加商品知识检索或购物Agent，不把RAG与后训练效果混为一谈。

## 上传GitHub前

模型权重、缓存、临时训练输出和环境变量已由`.gitignore`排除；会保留体积很小的基线评测结果。仓库不包含API Key或密码。

运行发布前检查：

```powershell
python -m unittest discover -s tests -v
python -m pip check
```

本项目暂未选择开源许可证；公开仓库可以展示和阅读，但如需明确允许他人复制、修改或再发布，应由仓库所有者选择并添加LICENSE。

## 项目目录

```text
data/
  processed/    清洗、划分后的训练数据
  eval/         固定业务评测集
src/            训练、评测和推理服务代码
scripts/        数据生成、训练、评测和部署脚本
tests/          单元测试
docs/           部署与QLoRA运行文档
```

## 硬件约定

本机负责开发和小规模验证；QLoRA/DPO 正式训练使用 NVIDIA GPU 环境。代码会保持同一套目录和配置，避免在本机与云端之间反复修改。
