import unittest

from pydantic import ValidationError

from src.ecommerce_llm.schemas import ChatRequest


class ChatRequestTests(unittest.TestCase):
    def test_valid_request(self) -> None:
        request = ChatRequest(messages=[{"role": "user", "content": "请推荐商品"}])
        self.assertEqual(request.max_new_tokens, 256)
        self.assertFalse(request.do_sample)

    def test_requires_user_message(self) -> None:
        with self.assertRaises(ValidationError):
            ChatRequest(messages=[{"role": "system", "content": "规则"}])

    def test_generation_length_has_upper_bound(self) -> None:
        with self.assertRaises(ValidationError):
            ChatRequest(
                messages=[{"role": "user", "content": "测试"}],
                max_new_tokens=2048,
            )


if __name__ == "__main__":
    unittest.main()

