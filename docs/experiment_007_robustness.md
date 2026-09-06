# 实验007：困难集与鲁棒性增强

## 改进内容

- 支持自由中文商品名称；
- 将克转换为千克后比较；
- 支持主排序与多个并列排序字段；
- 支持上下限冲突、无合适商品与字段缺失；
- 不可稳定解析时返回错误，不静默生成决策。

## 困难集

`ecommerce_robustness_v1.jsonl`共35题，包含自由名称、单位换算、阈值过滤、多级排序、无合适商品、字段缺失和冲突条件7类，每类5题。数据由人工定义场景后参数化生成，不能描述成35条完全独立人工语料。

## 结果

35/35题通过，原16题抽取评测和40题混合评测均未退化；自动化测试19项全部通过。

```powershell
python scripts\generate_robustness_cases.py
python scripts\evaluate_robustness.py
python -m unittest discover -s tests -v
```

当前结果证明受支持语法的确定性能力，不代表开放域语言理解。下一步加入规则失败时的大模型JSON抽取回退，并继续用Schema校验阻止不可审计输出进入决策引擎。
