# 实验007：困难集与鲁棒性增强

## 改进内容

- 支持自由中文商品名称；
- 将克转换为千克后比较；
- 支持主排序与多个并列排序字段；
- 支持上下限冲突、无合适商品与字段缺失；
- 不可稳定解析时返回错误，不静默生成决策。

## 困难集

`ecommerce_robustness_v1.jsonl`共 35 题，包含自由名称、单位换算、阈值过滤、多级排序、无合适商品、字段缺失和冲突条件共 7 类，每类 5 题；场景由人工定义后参数化生成，覆盖各类边界的系统性回归。

## 结果

35/35 题通过，原 16 题抽取评测与 40 题混合评测均未退化；自动化测试全部通过。

```powershell
python scripts\generate_robustness_cases.py
python scripts\evaluate_robustness.py
python -m unittest discover -s tests -v
```

该困难集验证了受支持语法范围内确定性解析的稳定性。下一步加入规则失败时的大模型 JSON 抽取回退（实验008 已落地），并继续用 Schema 校验阻止不可审计输出进入决策引擎。
