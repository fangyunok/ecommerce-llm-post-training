# 本地部署与推理参数调试

## 1. 当前部署结果

- 模型：Qwen2.5-0.5B-Instruct（0.49B参数）
- 下载目录：`models/Qwen2.5-0.5B-Instruct`
- 推理框架：PyTorch + Transformers
- API框架：FastAPI + Uvicorn
- 当前设备：CPU / float32
- API文档：`http://127.0.0.1:8000/docs`
- 健康检查：`http://127.0.0.1:8000/health`

启动：

```powershell
cd D:\面试项目
powershell -ExecutionPolicy Bypass -File scripts\start_api.ps1
```

测试：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\smoke_test.ps1
```

停止：在启动服务的终端按 `Ctrl+C`。

## 2. 请求链路

```text
JSON请求
  → Pydantic参数校验
  → 自动补充system prompt
  → 模型chat template
  → tokenizer生成input_ids
  → model.generate
  → 只截取新增token
  → decode为文本
  → 返回答案、token数和耗时
```

核心代码：

- `src/ecommerce_llm/app.py`：HTTP路由与错误处理。
- `src/ecommerce_llm/schemas.py`：请求、响应及参数边界。
- `src/ecommerce_llm/model_service.py`：模型加载、模板和生成。
- `src/ecommerce_llm/config.py`：环境变量配置。

## 3. 模型加载参数

### dtype

本机CPU使用 `torch.float32`，兼容性最好但内存较高。后续NVIDIA GPU优先使用：

- 支持BF16的GPU：`torch.bfloat16`；
- 不支持BF16但支持FP16的GPU：`torch.float16`；
- QLoRA训练：基座权重4-bit加载，计算dtype通常选BF16。

不要把“4-bit加载”与“模型所有计算都使用4-bit”混为一谈。QLoRA会以4-bit存储基座权重，但部分计算和LoRA参数仍使用更高精度。

### eval与inference_mode

- `model.eval()`：关闭dropout等训练行为；
- `torch.inference_mode()`：不构建反向传播计算图，减少内存和开销。

### use_cache

`use_cache=True`保存历史token的K/V缓存。自回归生成时不必每一步重复计算全部历史，通常能显著加速，但上下文越长，KV Cache占用越大。

## 4. 生成参数调试

### max_new_tokens

限制新生成token数量，不包含输入长度。

- 商品推荐短回答：`128–256`；
- 评论总结：`256–512`；
- 结构化长报告：`512–1024`。

过小会截断答案，过大会提高延迟，并可能产生冗余内容。服务当前上限为1024。

### do_sample

- `false`：贪心解码，每步选择最高概率token，可复现；
- `true`：按概率采样，多次请求可能产生不同回答。

商品参数核对、约束判断和离线评测建议先用 `false`。营销文案等需要多样性的任务再开启采样。

### temperature

仅在 `do_sample=true` 时生效：

- `0.2–0.5`：更保守；
- `0.6–0.8`：平衡；
- `>1.0`：随机性明显增大，事实任务风险更高。

它改变概率分布的尖锐程度，不会给模型补充新知识。

### top_p

仅在采样时生效。它保留累计概率达到阈值的最小候选集合：

- 事实型电商任务：`0.8–0.9`；
- 创意文案：`0.9–0.95`。

调试时不要同时大幅改变temperature和top_p，否则难以判断改善来自哪个参数。

### repetition_penalty

- `1.0`：不惩罚；
- `1.03–1.10`：轻微减少重复；
- 太高：可能破坏正常术语、数字和句式。

当前默认值为1.05。只有观察到明显复读后再继续上调。

## 5. 推荐的调参方法

固定一条个代表性评测问题，每次只改一个参数，并记录：

- 约束满足率；
- 商品参数准确率；
- 是否编造；
- 格式合规率；
- 平均输出时；
- 平均输出token数。

第一组基线采用确定性配置：

```json
{
  "max_new_tokens": 256,
  "do_sample": false,
  "repetition_penalty": 1.05
}
```

第二组测试可控采样：

```json
{
  "max_new_tokens": 256,
  "do_sample": true,
  "temperature": 0.5,
  "top_p": 0.9,
  "repetition_penalty": 1.05
}
```

## 6. 并发与生产注意事项

当前服务使用生成锁串行处理请求，优点是逻辑稳定、内存可控，适合本地演示；缺点是并发请求需要排队。

不要在当前机器上简单增加多个Uvicorn worker：每个worker都会各自加载一份模型，内存占用近似成倍增长。真正的GPU生产部署可切换到vLLM等支持连续批处理的推理引擎，并加入：

- 队列与超时；
- 流式输出；
- 鉴权与限流；
- 结构化日志和监控；
- 动态批处理；
- 容器化和健康探针。

## 7. 接入QLoRA模型

后续训练产出的LoRA Adapter目录通常包含adapter配置和权重。无需改API，只设置：

```powershell
$env:ADAPTER_PATH = "D:\面试项目\outputs\sft_adapter"
powershell -ExecutionPolicy Bypass -File scripts\start_api.ps1
```

加载过程是：

```text
同一个基座模型 + LoRA Adapter → 微调后的推理模型
```

必须保证Adapter对应的基座模型和训练时完全一致。

## 8. 当前baseline发现

测试条件要求预算250元且重视续航：

- A款：299元、30小时；
- B款：239元、40小时。

基座模型却推荐A款，违反预算且理由与给定数据矛盾。这不是部署失败，而是可复现的模型能力失败。后续SFT的目标之一是提升“硬约束满足率”，并用固定评测集验证改善，而不是只观察训练loss。

