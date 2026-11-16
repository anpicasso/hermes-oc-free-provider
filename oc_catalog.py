"""Policy over the official OpenCode catalog, never an alternate model registry.

Discovery retains unsupported transports with a reason. Inference always checks
fresh native metadata; an old picker/cache entry is not a free-model entitlement.
"""
from __future__ import annotations

import copy

SUPPORTED_PACKAGE = "@opencode/ai/providers/openai-compatible"


class CatalogError(ValueError):
    """A model cannot safely be selected by this free-only provider."""


def _free(cost) -> bool:
    if not isinstance(cost, list) or not cost:
        return False
    for tier in cost:
        if not isinstance(tier, dict) or not isinstance(tier.get("cache"), dict):
            return False
        prices = [tier.get("input"), tier.get("output"),
                  tier["cache"].get("read"), tier["cache"].get("write")]
        if any(type(value) not in (int, float) or value != 0 for value in prices):
            return False
    return True


def describe_models(records: list[dict]) -> list[dict]:
    """Return non-mutating annotated records; malformed/ambiguous catalogs fail."""
    if not isinstance(records, list):
        raise CatalogError("native model discovery failed: expected a catalog list")
    result, seen = [], set()
    for record in records:
        if not isinstance(record, dict):
            raise CatalogError("invalid native model record")
        provider, model = record.get("providerID"), record.get("id")
        if any(not isinstance(value, str) or not value or value != value.strip()
               for value in (provider, model)):
            raise CatalogError("invalid native model identity")
        key = (provider, model)
        if key in seen:
            raise CatalogError("duplicate native model identity")
        seen.add(key)
        row = copy.deepcopy(record)
        row["qualified_id"] = f"{provider}/{model}"
        row["free_eligible"] = (provider == "opencode" and record.get("enabled") is True
                                and _free(record.get("cost")))
        package = record.get("package") or record.get("providerPackage")
        reason = None
        if record.get("enabled") is not True:
            reason = "model is not enabled in the isolated native catalog"
        elif record.get("type", "chat") != "chat":
            reason = "unsupported non-chat model type"
        elif package != SUPPORTED_PACKAGE:
            reason = f"unsupported transport package: {package!r}; Chat Completions SSE required"
        elif record.get("status") == "deprecated":
            reason = "model is deprecated in the native catalog"
        row["supported"] = reason is None
        row["unsupported_reason"] = reason
        result.append(row)
    return result


def free_model_ids(records: list[dict]) -> list[str]:
    # Unsupported protocols stay visible; selection explains why they cannot run.
    return [row["id"] for row in describe_models(records) if row["free_eligible"]]


def select_model(records: list[dict], model: str) -> dict:
    if not isinstance(model, str) or not model or model != model.strip():
        raise CatalogError("an exact native model ID is required")
    for row in describe_models(records):
        if model not in (row["id"], row["qualified_id"]):
            continue
        if row["providerID"] != "opencode":
            continue
        if not row["free_eligible"]:
            raise CatalogError("free-only policy: disabled, missing, unknown or nonzero native pricing")
        if not row["supported"]:
            raise CatalogError(row["unsupported_reason"])
        return row
    raise CatalogError(f"model {model!r} is absent from the native OpenCode free catalog")
