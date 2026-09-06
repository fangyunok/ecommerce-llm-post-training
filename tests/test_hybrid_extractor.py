import unittest

from src.ecommerce_llm.hybrid_extractor import HybridConstraintExtractor, ground_constraint_operators, normalize_numeric_values, parse_json_object
from src.ecommerce_llm.schemas import ChatMessage, ChatResponse, Usage


class FakeGenerator:
    def __init__(self, content):
        self.content = content
        self.calls = 0

    def generate(self, _request):
        self.calls += 1
        return ChatResponse(model="fake", message=ChatMessage(role="assistant", content=self.content), usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2), latency_seconds=0)


class HybridExtractorTests(unittest.TestCase):
    def test_rules_skip_model(self):
        generator = FakeGenerator("invalid")
        request, trace = HybridConstraintExtractor(generator, True).extract("A款100元；B款90元。选择价格最低的。")
        self.assertEqual((request.products[1].name, trace.source, generator.calls), ("B款", "deterministic_parser", 0))

    def test_model_fallback_and_markdown_json(self):
        payload = '```json\n{"products":[{"name":"红色款","values":{"price":99}}],"constraints":[],"sort":{"field":"price","direction":"asc"}}\n```'
        generator = FakeGenerator(payload)
        request, trace = HybridConstraintExtractor(generator, True).extract("开放式复杂描述")
        self.assertEqual((request.products[0].name, trace.source, generator.calls), ("红色款", "llm_fallback", 1))

    def test_invalid_model_output_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "Schema校验"):
            HybridConstraintExtractor(FakeGenerator("不是JSON"), True).extract("开放式复杂描述")

    def test_disabled_fallback_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "未启用"):
            HybridConstraintExtractor(FakeGenerator("{}"), False).extract("开放式复杂描述")

    def test_json_can_be_surrounded_by_text(self):
        self.assertEqual(parse_json_object('结果：{"a":1} 完成'), {"a": 1})

    def test_common_units_are_normalized_before_schema_validation(self):
        payload = {"products": [{"name": "A", "values": {"camera": "64MP", "weight": "1.4kg"}}],
                   "constraints": [{"field": "camera", "operator": "ge", "value": "48MP"}]}
        normalized = normalize_numeric_values(payload)
        self.assertEqual(normalized["products"][0]["values"], {"camera": 64.0, "weight": 1.4})
        self.assertEqual(normalized["constraints"][0]["value"], 48.0)

    def test_explicit_source_phrase_grounds_operator(self):
        payload = {"constraints": [{"field": "weight", "operator": "ge", "value": 1.5}]}
        grounded = ground_constraint_operators("绝不能重于1.5kg", payload)
        self.assertEqual(grounded["constraints"][0]["operator"], "le")


if __name__ == "__main__":
    unittest.main()
