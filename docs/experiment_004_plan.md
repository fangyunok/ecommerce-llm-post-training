# 实验004计划：1.5B模型规模对照

## 实验问题

在训练数据、QLoRA参数和评测集不变时，将基座从`Qwen2.5-0.5B-Instruct`扩大到`Qwen2.5-1.5B-Instruct`，能否改善数字比较、属性绑定和拒答能力？

## 控制变量

- 训练数据：实验002的1800/225条v2数据；
- 评测集：同一40题`ecommerce_eval_v2.jsonl`；
- 训练：3 epochs、学习率2e-4、batch size 2、梯度累积8；
- LoRA：r=16、alpha=32、dropout=0.05；
- 推理：贪心解码、max_new_tokens 160、repetition_penalty 1.05。

唯一主要变量是模型从0.5B变为1.5B。

## 执行顺序

1. 不加载Adapter，评测1.5B基座并保存为`base_1_5b_extended.jsonl`。
2. 使用v2数据训练，Adapter保存为`outputs/sft_adapter_1_5b_v2`。
3. 加载新Adapter复测40题，保存为`sft_1_5b_v2_extended.jsonl`。
4. 比较1.5B微调前后，以及0.5B最佳实验002与1.5B微调结果。

## 成功标准

- 1.5B微调后总体准确率高于0.5B实验002的57.5%；
- 数值筛选、预算约束、信息不足和无合适商品至少三个类别改善；
- 记录模型规模带来的训练时长、显存、推理延迟和Adapter体积变化；
- 若效果没有提升，也保留结果并分析模型规模不是唯一瓶颈。
