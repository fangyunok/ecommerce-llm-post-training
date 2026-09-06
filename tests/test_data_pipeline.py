import unittest
from collections import Counter
from pathlib import Path

from scripts.generate_sft_data import GENERATORS, build_split, validate
from src.evaluation.evaluate_api import load_jsonl, validate_cases


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class TrainingDataTests(unittest.TestCase):
    def test_generated_split_is_unique_and_balanced(self) -> None:
        rows = build_split(180, seed=123)
        validate(rows)
        prompts = [row["prompt"][-1]["content"] for row in rows]
        counts = Counter(row["category"] for row in rows)

        self.assertEqual(len(prompts), len(set(prompts)))
        self.assertEqual(len(counts), len(GENERATORS))
        self.assertTrue(all(count == 20 for count in counts.values()))

    def test_train_and_validation_have_no_exact_prompt_overlap(self) -> None:
        train = build_split(180, seed=123)
        validation = build_split(90, seed=124)
        train_prompts = {row["prompt"][-1]["content"] for row in train}
        validation_prompts = {row["prompt"][-1]["content"] for row in validation}
        self.assertFalse(train_prompts & validation_prompts)


class EvaluationDataTests(unittest.TestCase):
    def test_extended_evaluation_set_is_valid(self) -> None:
        path = PROJECT_ROOT / "data" / "eval" / "ecommerce_eval_v2.jsonl"
        cases = load_jsonl(path)
        self.assertEqual(len(cases), 40)
        self.assertGreaterEqual(len({case["category"] for case in cases}), 6)

    def test_duplicate_ids_are_rejected(self) -> None:
        case = {
            "id": "duplicate",
            "category": "test",
            "messages": [{"role": "user", "content": "test"}],
            "checks": {"all_of": [], "any_of": [], "none_of": []},
        }
        with self.assertRaisesRegex(ValueError, "重复"):
            validate_cases([case, case])


if __name__ == "__main__":
    unittest.main()
