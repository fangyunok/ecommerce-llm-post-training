# 大模型后训练与结构化输出服务

[![tests](https://github.com/fangyunok/ecommerce-llm-post-training/actions/workflows/ci.yml/badge.svg)](https://github.com/fangyunok/ecommerce-llm-post-training/actions/workflows/ci.yml)

**可复现的大模型后训练与结构化输出服务：Prompt 约束 → JSON 抽取 → Schema 校验 → 确定性决策 → 服务化交付。**

项目回答两个问题：**小模型后训练能改善哪些任务**，以及**哪些确定性业务约束不应该交给大模型猜测**。核心结论是：语言任务交给模型，数值与排序决策交回规则，两者通过受 JSON Schema 约束的结构化输出连接。

完整链路：

`业务定义 → 数据构造 → 基座评测 → QLoRA/SFT → 结构化输出与 Prompt 工程 → 规则—模型协同 → 推理服务`

## 先看结论

| 子系统 | 对照结果 | 结论与边界 |
|---|---|---|
| **模型后训练（SFT / QLoRA）** | 1.5B 固定 40 题集 `18/40 → 24/40`；0.5B 峰值 `23/40` | 扩模与 SFT 均有稳定增益，但不足以可靠执行硬数值约束 |
| **为什么不能只靠扩模** | 数值筛选题仅 `1/11` | 模型擅长语言表达，不适合做确定性数值判断；继续堆数据与参数不是最优解 |
| **Prompt 工程 + 结构化输出** | 自然语言 → JSON 约束抽取，`16/16` 字段与决策全通过（原题 `11/11` + 独立改写 `5/5`） | 抽取表达范围受限于受支持的 schema，非开放域 |
| **规则—模型协同** | 混合架构 `34/40`（`85.0%`），较 1.5B SFT `60.0%` 提升 25 个百分点 | 数值题 `1/11 → 11/11`；该口径为离线严格准确率，不是线上指标 |
| **Schema 校验与失败封闭** | Pydantic 二次校验，校验失败返回 `422` | 未经验证的结构化数据不进入规则决策 |
| **鲁棒性与回归** | 7 类 35 题参数化困难集 `35/35`；真实 1.5B JSON 回退 `5/5`；统一接口三路由 `3/3` | 困难集为参数化构造，5 题回退集只用于链路验收，均不等于线上分布 |
| **服务化与工程交付** | FastAPI 统一接口 + Docker + GitHub Actions（30 项单元测试） | 无 GPU 也能验证规则、抽取与评测链路 |

> 这些结果用于证明实验与工程闭环，不代表线上业务指标。固定集和合成数据的规模、模板偏差及适用边界均在各实验报告中单独披露。

## 系统架构

```mermaid
flowchart LR
    A[用户请求] --> B[统一购物助手 API]
    B --> C{意图与约束路由}
    C -->|数值筛选/排序| D[确定性规则引擎]
    C -->|自然语言约束| E[Schema 抽取]
    E --> F[单位归一化与原文校验]
    F -->|校验通过| D
    F -->|无法可靠结构化| G[失败封闭或 LLM 回退]
    C -->|语言生成| G
    D --> H[可审计决策]
    G --> I[模型回答]
```

核心原则是**把概率能力与确定性能力分离**：模型负责语言理解与表达，Python 规则负责预算、阈值和排序。响应会暴露实际路由、结构化抽取结果与候选集合，便于定位错误发生在抽取、规则还是生成阶段。

这个仓库聚焦**后训练、Prompt 与结构化输出、规则—模型协同**；不把检索系统、通用 Agent 编排或纯推荐模型训练混在同一个项目里。

## 能力边界与可迁移性

- **与任务无关、可直接复用**：结构化输出协议层（Prompt 约束 → JSON 解析 → Pydantic Schema 二次校验 → 失败封闭）、规则—模型协同路由层（概率能力与确定性能力分离）、分层评测框架（严格准确率、字段级抽取准确率、困难集回归、路由验收）。
- **与本任务绑定**：电商商品字段 schema、预算 / 容量 / 尺寸等领域约束定义、固定业务评测集。
- **迁移到新的业务场景时**，只需替换后两者，结构化输出协议层、协同路由层与评测协议可直接沿用。

## 项目状态

### 训练与评测

| 模块 | 状态 | 结果 |
|---|---|---|
| 本地 CPU 推理 API | 已完成 | FastAPI 健康检查与聊天接口可用 |
| 固定业务评测（8 题） | 已完成 | 基座严格准确率 `2/8`（25%），首轮 SFT 后 `4/8`（50%） |
| SFT 数据 v2 | 已完成 | 1800 条训练、225 条验证，9 类均衡分布 |
| 扩展业务评测（40 题） | 已完成 | 实验001 `17/40`（42.5%），实验002 `23/40`（57.5%） |
| 数据质量对照实验 | 已完成 | 否定、数值筛选和摘要提升，但预算与拒答退化 |
| 实验003 恢复测试 | 已完成 | `21/40`（52.5%），恢复拒答但总体低于实验002 |
| 1.5B 模型规模对照 | 已完成 | 基座 `18/40`（45.0%），SFT 后 `24/40`（60.0%） |
| 规则与模型协同 | 已完成第一版 | 11 道数值题规则路由全通过，混合结果 `34/40`（85.0%） |
| 自然语言约束抽取 | 已完成第一版 | 原题 `11/11`、独立改写题 `5/5`，字段与决策均全通过 |
| 困难集与鲁棒性 | 已完成 | 35 题、7 类参数化困难集全部通过，19 项测试通过 |
| 规则优先与 LLM 回退 | 已完成代码与模拟验证 | 规则成功零模型调用，Schema 失败封闭；待 GPU 真实模型验证 |
| 真实 1.5B 抽取回退 | 已完成 | 首轮 `3/5`，加入单位归一化与原文约束校验后 `5/5`；统一接口 `3/3` |
| QLoRA/SFT 首轮训练 | 已完成 | RTX 3090 训练约 11 分 13 秒，Adapter 已保存 |

### 服务与交付

| 模块 | 状态 | 结果 |
|---|---|---|
| 统一购物助手接口 | 已完成 | `/v1/shopping-assistant` 统一决策、抽取回退与语言任务 |
| Docker 与 CI | 已完成并验证 | GitHub Actions 单测与生产镜像构建成功，总耗时 2 分 13 秒 |

### 后续阶段

- **DPO 偏好对齐**：未开始，仅在新增偏好数据质量可控且有独立测试集时开展
- **RAG 与购物 Agent**：未开始，不用于替代本仓库的后训练对照结论

> 首轮结果证明训练与部署闭环可运行，但 8 条固定评测集规模较小，合成训练集模板规律较强；50% 不能解释为真实线上业务准确率。

## 实验索引

实验002的完整对照、分类变化和局限性见[实验002报告](docs/experiment_002_qlora.md)。
实验003的数据设计和停止条件见[实验003计划](docs/experiment_003_plan.md)。
实验003结果见[实验003报告](docs/experiment_003_qlora.md)。
1.5B模型规模对照见[实验004计划](docs/experiment_004_plan.md)。
实验004的完整结果、分类变化和局限性见[实验004报告](docs/experiment_004_qlora.md)。
规则与模型协同的设计、评测口径和边界见[实验005报告](docs/experiment_005_hybrid_rules.md)。
自然语言结构化抽取的实现与泛化测试见[实验006报告](docs/experiment_006_extraction.md)。
困难集、单位归一化和多级排序结果见[实验007报告](docs/experiment_007_robustness.md)。
真实1.5B模型抽取回退及统一接口验收见[实验008报告](docs/experiment_008_llm_fallback.md)。
首轮QLoRA/SFT闭环见[实验001报告](docs/experiment_001_qlora.md)。

最终系统架构见[架构说明](docs/architecture.md)，部署步骤见[生产部署手册](docs/production_deployment.md)，项目设计与结果总结见[项目讲解材料](docs/interview_materials.md)。

## 5分钟验证核心能力

无需下载模型即可运行规则、抽取和鲁棒性回归：

```powershell
python -m pip install -r requirements-test.txt
python -m unittest discover -s tests -v
python scripts\generate_robustness_cases.py
python scripts\evaluate_robustness.py
```

需要体验完整 API 时，再安装推理依赖并启动服务：

```powershell
python -m pip install -r requirements-serving.txt
powershell -ExecutionPolicy Bypass -File scripts\start_api.ps1
```

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
- `POST /v1/rule-recommendations`：接收结构化商品、硬约束和排序字段，返回可审计的确定性决策。
- `POST /v1/natural-language-recommendations`：将受支持的自然语言商品比较请求抽取成JSON，再调用规则引擎决策。
- `POST /v1/shopping-assistant`：统一业务入口；支持`auto/decision/chat`模式并返回实际路由。

若后续已有LoRA Adapter，可设置 `ADAPTER_PATH` 后复用同一服务。

### 运行固定基线评测

保持API运行，在另一个PowerShell执行：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\evaluate_baseline.ps1
```

评测结果保存到 `outputs/evaluation/baseline.jsonl`，后续微调模型使用同一评测集对比。

当前基座评测结果：`2/8`严格通过，准确率`25%`。首轮SFT后为`4/8`，准确率`50%`。完整记录见[首轮QLoRA实验报告](docs/experiment_001_qlora.md)。

### 生成SFT数据

```powershell
python scripts\generate_sft_data.py
```

默认生成1800条训练数据和225条验证数据，格式为对话式prompt/completion。v2重点增加预算边界、否定属性、数值字段绑定和多商品筛选。

### 在NVIDIA GPU环境进行QLoRA

```powershell
python -m pip install -r requirements-training.txt
powershell -ExecutionPolicy Bypass -File scripts\train_qlora.ps1
```

本机没有CUDA，训练命令应在Colab、AutoDL或实验室GPU服务器执行。首轮实验已在AutoDL RTX 3090上完成；训练完成后可将`outputs/sft_adapter`作为`ADAPTER_PATH`接入现有API。

训练前先阅读并执行[QLoRA运行手册](docs/qlora_runbook.md)中的环境体检与参数检查。

## 实验演进

1. **实验001：跑通闭环。** 完成首轮QLoRA/SFT与固定集评测，严格准确率由2/8提升到4/8。
2. **实验002—003：数据质量对照。** 扩展到40题，定位否定、数值筛选、预算和拒答之间的权衡，并记录一次未带来总体提升的恢复实验。
3. **实验004：模型规模对照。** 1.5B基座18/40，SFT后24/40，确认扩模有效但不足以可靠执行硬约束。
4. **实验005—007：改变系统设计。** 将数值问题交给规则引擎，补充自然语言抽取、单位归一化、多级排序和35题困难集。
5. **实验008：验证真实模型回退。** 1.5B抽取首轮3/5，经约束校验后5/5；统一接口验收3/3。

运行规则与模型协同评测：

实验005复用实验004保存的1.5B模型输出，并将11道数值筛选题路由到确定性规则引擎：

```powershell
python scripts\evaluate_hybrid.py
python scripts\evaluate_extraction.py
python scripts\evaluate_llm_fallback.py
```

这两项评测都不加载大模型、不需要GPU。85.0%的混合结果使用已结构化JSON；实验006另行验证了受支持自然语言的自动抽取，但其表达范围仍有限，不能将结果直接解释为开放域线上准确率。

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

## 安全与复现

- 模型权重、缓存、临时训练输出和环境变量已由 `.gitignore` 排除；会保留体积很小的基线评测结果。
- 仓库不包含 API Key 或密码。
- 发布前运行检查：

```powershell
python -m unittest discover -s tests -v
python -m pip check
```

## License

MIT，详见 [LICENSE](LICENSE)。
