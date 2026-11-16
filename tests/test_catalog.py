"""Native catalog policy: no inferred prices, renamed identifiers or transport guesses."""
import copy
import unittest

from oc_catalog import (CatalogError, SUPPORTED_PACKAGE, describe_models,
                        free_model_ids, select_model)


def model(name="new-free", **overrides):
    return {
        "id": name, "modelID": name, "providerID": "opencode", "enabled": True,
        "name": name, "status": "active", "capabilities": {"tools": True},
        "providerPackage": "@opencode/ai/providers/openai-compatible",
        "cost": [{"input": 0, "output": 0, "cache": {"read": 0, "write": 0}}],
        **overrides,
    }


class CatalogTests(unittest.TestCase):
    def test_exact_selection_and_no_alias_repair(self):
        rows = [model()]
        self.assertEqual(select_model(rows, "new-free")["id"], "new-free")
        self.assertEqual(select_model(rows, "opencode/new-free")["id"], "new-free")
        for invalid in (" new-free", "new-free ", "New-Free", "other/new-free", "new-free#fast"):
            with self.subTest(invalid=invalid), self.assertRaises(CatalogError):
                select_model(rows, invalid)

    def test_free_requires_every_price_in_every_tier(self):
        for cost in ([], None, {}, [{"input": 0, "output": 0}],
                     [{"input": False, "output": 0, "cache": {"read": 0, "write": 0}}],
                     [{"input": 0, "output": 0, "cache": {"read": 0, "write": .01}}]):
            with self.subTest(cost=cost):
                self.assertEqual(free_model_ids([model(cost=cost)]), [])
                with self.assertRaises(CatalogError):
                    select_model([model(cost=cost)], "new-free")
        tiered = model()
        tiered["cost"].append({"tier": {"type": "context", "size": 200000},
                               "input": 0, "output": 1, "cache": {"read": 0, "write": 0}})
        self.assertEqual(free_model_ids([tiered]), [])

    def test_unsupported_protocol_remains_discoverable(self):
        row = model(package="@opencode/ai/providers/anthropic")
        self.assertEqual(free_model_ids([row]), ["new-free"])
        described = describe_models([row])[0]
        self.assertFalse(described["supported"])
        self.assertIn("transport", described["unsupported_reason"])
        with self.assertRaisesRegex(CatalogError, "transport"):
            select_model([row], "new-free")

    def test_model_package_overrides_provider_package(self):
        inherited = model(package=None)
        self.assertTrue(describe_models([inherited])[0]["supported"])
        overridden = model(package=SUPPORTED_PACKAGE,
                           providerPackage="@opencode/ai/providers/anthropic")
        self.assertTrue(describe_models([overridden])[0]["supported"])
        blocked = model(package="@opencode/ai/providers/anthropic",
                        providerPackage=SUPPORTED_PACKAGE)
        self.assertFalse(describe_models([blocked])[0]["supported"])
        self.assertIn("transport", describe_models([blocked])[0]["unsupported_reason"])

    def test_disabled_non_chat_and_deprecated_keep_their_reasons(self):
        for row, expected in ((model(type="image"), "non-chat"),
                              (model(status="deprecated"), "deprecated"),
                              (model(enabled=False), "not enabled")):
            with self.subTest(expected=expected):
                described = describe_models([row])[0]
                self.assertFalse(described["supported"])
                self.assertIn(expected, described["unsupported_reason"])

    def test_inventory_preserved_and_input_not_mutated(self):
        rows = [model(), model("paid", cost=[]), model("disabled", enabled=False)]
        before = copy.deepcopy(rows)
        self.assertEqual(len(describe_models(rows)), 3)
        self.assertEqual(rows, before)
        self.assertEqual(free_model_ids(rows), ["new-free"])

    def test_empty_differs_from_failed_and_duplicate_rejected(self):
        self.assertEqual(free_model_ids([]), [])
        with self.assertRaises(CatalogError):
            free_model_ids(None)
        with self.assertRaises(CatalogError):
            describe_models([model(), model()])

    def test_paid_provider_cannot_enter_free_route(self):
        row = model(providerID="other")
        self.assertEqual(free_model_ids([row]), [])
        with self.assertRaises(CatalogError):
            select_model([row], "other/new-free")

    def test_invalid_metadata_and_negative_prices_fail_closed(self):
        for row in (model(id=""), model(enabled="true"), model(cost=[{"input": -1, "output": 0, "cache": {"read": 0, "write": 0}}])):
            with self.subTest(row=row), self.assertRaises(CatalogError):
                select_model([row], "new-free")


if __name__ == "__main__":
    unittest.main()
