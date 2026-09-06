import unittest

from src.ecommerce_llm.constraint_extractor import extract_rule_request
from src.ecommerce_llm.recommendation_engine import recommend_by_rules


class ExtractorRobustnessTests(unittest.TestCase):
    def test_free_product_names_and_gram_conversion(self):
        request = extract_rule_request(
            "星河本价格4999元、重1350g；云帆本价格5199元、重1.2kg。"
            "预算不超过5100元，重量不高于1.4kg，优先低价。"
        )
        self.assertEqual(request.products[0].name, "星河本")
        self.assertEqual(request.products[0].values["weight"], 1.35)
        self.assertEqual(recommend_by_rules(request).selected_product, "星河本")
