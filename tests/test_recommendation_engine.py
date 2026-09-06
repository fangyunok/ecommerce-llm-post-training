import unittest

from src.ecommerce_llm.recommendation_engine import recommend_by_rules
from src.ecommerce_llm.schemas import RuleRecommendationRequest


class RecommendationEngineTests(unittest.TestCase):
    def test_filters_before_sorting(self):
        request = RuleRecommendationRequest.model_validate(
            {
                "products": [
                    {"name": "A", "values": {"price": 99, "capacity": 64}},
                    {"name": "B", "values": {"price": 109, "capacity": 128}},
                    {"name": "C", "values": {"price": 119, "capacity": 256}},
                ],
                "constraints": [{"field": "capacity", "operator": "ge", "value": 128}],
                "sort": {"field": "price", "direction": "asc"},
            }
        )
        result = recommend_by_rules(request)
        self.assertEqual(result.selected_product, "B")
        self.assertEqual(result.eligible_products, ["B", "C"])

    def test_budget_boundary_is_inclusive(self):
        request = RuleRecommendationRequest.model_validate(
            {
                "products": [
                    {"name": "M", "values": {"price": 300, "battery": 80}},
                    {"name": "N", "values": {"price": 301, "battery": 120}},
                ],
                "constraints": [{"field": "price", "operator": "le", "value": 300}],
                "sort": {"field": "battery", "direction": "desc"},
            }
        )
        self.assertEqual(recommend_by_rules(request).selected_product, "M")

    def test_reports_missing_sort_field(self):
        request = RuleRecommendationRequest.model_validate(
            {
                "products": [{"name": "A", "values": {"price": 100}}],
                "sort": {"field": "score", "direction": "desc"},
            }
        )
        result = recommend_by_rules(request)
        self.assertIsNone(result.selected_product)
        self.assertIn("信息不足", result.answer)

    def test_reports_no_match(self):
        request = RuleRecommendationRequest.model_validate(
            {
                "products": [{"name": "A", "values": {"price": 101}}],
                "constraints": [{"field": "price", "operator": "le", "value": 100}],
                "sort": {"field": "price", "direction": "asc"},
            }
        )
        result = recommend_by_rules(request)
        self.assertIsNone(result.selected_product)
        self.assertIn("没有合适商品", result.answer)


if __name__ == "__main__":
    unittest.main()
