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

## 学习路线

1. **数据与语言模型基础**：token、因果语言模型、聊天模板、SFT 数据格式。
2. **Transformer 最小原理**：Embedding、Self-Attention、因果掩码、前馈网络。
3. **基座模型评测**：建立训练前 baseline，确定业务指标。
4. **LoRA 与 QLoRA**：先理解低秩更新，再进行监督微调。
5. **SFT 实验**：记录超参数、训练曲线、显存与效果。
6. **DPO 偏好对齐**：构造 chosen/rejected 数据并完成对比实验。
7. **项目交付**：自动评测、错误分析、Gradio/FastAPI 演示和简历材料。

## 上传GitHub前

模型权重、缓存、临时训练输出和环境变量已由`.gitignore`排除；会保留体积很小的基线评测结果。仓库不包含API Key或密码。

运行发布前检查：

```powershell
python -m unittest discover -s tests -v
python -m pip check
```

本项目暂未选择开源许可证；公开仓库可以展示和阅读，但如需明确允许他人复制、修改或再发布，应由仓库所有者选择并添加LICENSE。

当前进度：**第 3 课——Self-Attention 与因果掩码**。

## 第 1 课运行方式

```powershell
python lessons/lesson01_data_anatomy.py
```

你应当观察三件事：

- 一条样本由哪些角色组成；
- 训练文本和推理提示为什么不一样；
- 为什么数据格式错误会直接影响训练质量。

## 第 2 课运行方式

```powershell
python lessons/lesson02_next_token.py
```

本课使用一个字符级极简模型展示完整链路。真实大模型的 tokenizer 和网络结构复杂得多，但监督信号仍然来自“将标签相对输入错开一位”。

## 第 3 课运行方式

```powershell
python lessons/lesson03_self_attention.py
```

本课从零实现单头 Self-Attention，观察 Q/K/V、缩放点积、Softmax 和因果掩码如何让同一个 token 在不同上下文中得到不同表示。

## 项目目录

```text
data/
  raw/          原始数据
  processed/    清洗、划分后的训练数据
lessons/        每一课的可运行练习
src/            后续加入正式训练、评测和推理代码
tests/          数据与代码测试
```

## 硬件约定

本机负责开发和小规模验证；QLoRA/DPO 正式训练使用 NVIDIA GPU 环境。代码会保持同一套目录和配置，避免在本机与云端之间反复修改。

