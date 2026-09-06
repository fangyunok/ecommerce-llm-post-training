import unittest

from src.ecommerce_llm.hybrid_extractor import HybridConstraintExtractor
from src.ecommerce_llm.schemas import ChatMessage, ChatResponse, ShoppingAssistantRequest, Usage
from src.ecommerce_llm.shopping_assistant import ShoppingAssistantService


class FakeGenerator:
    def __init__(self, content="模型回答"):
        self.content = content
        self.calls = 0

    def generate(self, _request):
        self.calls += 1
        return ChatResponse(model="fake", message=ChatMessage(role="assistant", content=self.content), usage=Usage(prompt_tokens=2, completion_tokens=2, total_tokens=4), latency_seconds=0)


class ShoppingAssistantTests(unittest.TestCase):
    def test_numeric_decision_uses_rules(self):
        generator = FakeGenerator()
        service = ShoppingAssistantService(HybridConstraintExtractor(generator, False), generator)
        result = service.handle(ShoppingAssistantRequest(text="A款100元；B款90元。选价格最低的。"))
        self.assertEqual((result.route, generator.calls), ("deterministic_rules", 0))

    def test_non_decision_uses_language_model(self):
        generator = FakeGenerator("已为你总结")
        service = ShoppingAssistantService(HybridConstraintExtractor(generator, False), generator)
        result = service.handle(ShoppingAssistantRequest(text="请总结这段商品介绍"))
        self.assertEqual((result.route, result.answer, generator.calls), ("language_model", "已为你总结", 1))

    def test_failed_decision_does_not_fall_through_to_chat(self):
        generator = FakeGenerator()
        service = ShoppingAssistantService(HybridConstraintExtractor(generator, False), generator)
        with self.assertRaises(ValueError):
            service.handle(ShoppingAssistantRequest(text="预算500元，请推荐一款好商品"))
        self.assertEqual(generator.calls, 0)

    def test_explicit_chat_mode_uses_model(self):
        generator = FakeGenerator()
        service = ShoppingAssistantService(HybridConstraintExtractor(generator, False), generator)
        result = service.handle(ShoppingAssistantRequest(text="预算是什么概念？", mode="chat"))
        self.assertEqual(result.route, "language_model")


if __name__ == "__main__":
    unittest.main()
