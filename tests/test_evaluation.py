import unittest

from src.evaluation.evaluate_api import check_answer


class EvaluationRuleTests(unittest.TestCase):
    def test_all_required_facts_pass(self) -> None:
        result = check_answer(
            "推荐B款，售价239元，续航40小时。",
            {"all_of": ["推荐B款", "239", "40"], "any_of": [], "none_of": []},
        )
        self.assertTrue(result.passed)

    def test_missing_required_fact_fails(self) -> None:
        result = check_answer(
            "推荐B款。",
            {"all_of": ["推荐B款", "239"], "any_of": [], "none_of": []},
        )
        self.assertFalse(result.passed)
        self.assertIn("包含:239", result.failed_checks)

    def test_any_and_forbidden_rules(self) -> None:
        passing = check_answer(
            "信息不足，无法判断。",
            {"all_of": [], "any_of": ["信息不足", "无法推荐"], "none_of": ["推荐A款"]},
        )
        failing = check_answer(
            "推荐A款。",
            {"all_of": [], "any_of": ["信息不足", "无法推荐"], "none_of": ["推荐A款"]},
        )
        self.assertTrue(passing.passed)
        self.assertFalse(failing.passed)


if __name__ == "__main__":
    unittest.main()

