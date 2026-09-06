import unittest

from src.ecommerce_llm.hybrid_extractor import HybridConstraintExtractor, parse_json_object
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


if __name__ == "__main__":
    unittest.main()
