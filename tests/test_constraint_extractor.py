import json
import unittest
from pathlib import Path

from src.ecommerce_llm.constraint_extractor import extract_rule_request
from src.ecommerce_llm.recommendation_engine import recommend_by_rules
from src.evaluation.evaluate_api import check_answer


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ConstraintExtractorTests(unittest.TestCase):
    def test_holdout_paraphrases_make_correct_decisions(self):
        path = PROJECT_ROOT / "data/eval/ecommerce_extraction_holdout.jsonl"
        cases = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        for case in cases:
            with self.subTest(case=case["id"]):
                request = extract_rule_request(case["text"])
                answer = recommend_by_rules(request).answer
                self.assertTrue(check_answer(answer, case["checks"]).passed, answer)

    def test_rejects_unstructured_unsupported_text(self):
        with self.assertRaises(ValueError):
            extract_rule_request("我想买一个性价比高的商品，请推荐")


if __name__ == "__main__":
    unittest.main()
