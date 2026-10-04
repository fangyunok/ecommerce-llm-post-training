# 生产部署手册

## 本地启动

复制`.env.example`为`.env`并按需设置模型及Adapter，然后执行：

```powershell
docker compose up --build -d
docker compose ps
```

验证：

```powershell
curl http://127.0.0.1:8000/live
curl http://127.0.0.1:8000/health
```

`/live`只检查进程存活，供容器编排使用；`/health`额外报告模型状态、设备和加载错误。

## 统一接口示例

```json
POST /v1/shopping-assistant
{
  "text": "星河电脑4999元、1.4kg；云帆电脑5299元、1.2kg。预算不超过5100元，重量不高于1.5kg，优先低价。",
  "mode": "auto"
}
```

每个响应带`X-Request-ID`，日志记录请求方法、路径、状态码和耗时。数字决策无法可靠抽取时返回422；不会自动降级为不受约束的模型推荐。

## GPU部署

设置`MODEL_ID=Qwen/Qwen2.5-1.5B-Instruct`、`ADAPTER_PATH`和`ENABLE_LLM_EXTRACTION_FALLBACK=true`。生产环境应将模型缓存和Adapter挂载为只读卷，并由基础设施提供NVIDIA Container Toolkit。

## 验证状态

- 已验证：Python 接口、存活检查、请求日志、30 项单元测试、35 题困难集；
- 已验证：Compose 与 Actions YAML 可解析；
- 已验证：GitHub Actions 生产镜像构建成功，Docker job 耗时 1 分 54 秒；容器运行验证由 GitHub Linux runner 完成；
- 已验证：LLM 抽取回退在真实 1.5B 模型上完成评测，固定困难集 `5/5`、统一接口三路由 `3/3`（详见[实验008](experiment_008_llm_fallback.md)）。

真实模型回退评测使用 5 道刻意无法被规则解析的开放表达：

```powershell
python scripts\evaluate_llm_fallback.py
```
