"""Hermes model-provider registration for OpenCode."""

# ruff: noqa: E402, I001 -- Hermes imports plugin directories without a package name.

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_PLUGIN_DIR = Path(__file__).parent
if str(_PLUGIN_DIR) not in sys.path:
    sys.path.insert(0, str(_PLUGIN_DIR))

try:
    # Relative import: Hermes loads the plugin directory as a package, so every
    # installed copy gets its own client module. A shared top-level module
    # would let a stale copy on disk serve this copy's code.
    from .hermes_oc_client import (  # type: ignore[import-not-found]
        DEFAULT_MODEL,
        FALLBACK_MODELS,
        LOGICAL_BASE_URL,
        OpenCodeClient,
        sync_profile_catalog,
    )
except ImportError:  # loaded as a flat module (no package context)
    from hermes_oc_client import (  # type: ignore[import-not-found]
        DEFAULT_MODEL,
        FALLBACK_MODELS,
        LOGICAL_BASE_URL,
        OpenCodeClient,
        sync_profile_catalog,
    )
from providers import register_provider  # type: ignore[import-not-found]
from providers.base import ProviderProfile  # type: ignore[import-not-found]


class OpenCodeFreeProfile(ProviderProfile):
    def create_client(self, **client_kwargs: Any) -> OpenCodeClient:
        return OpenCodeClient(**client_kwargs)

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 15.0,
    ) -> list[str] | None:
        client = OpenCodeClient(
            api_key=api_key,
            base_url=base_url or self.base_url,
        )
        try:
            models = client.list_models(timeout=timeout)
        except Exception:  # noqa: BLE001 -- catalog hooks are required to fail soft.
            return None
        finally:
            client.close()
        if not models:
            return None
        # Hermes merges the profile's curated floor with the live list, so the
        # floor has to follow the relay: a static one keeps retired ids in the
        # picker and hides newly launched free models.
        sync_profile_catalog(self, models)
        return list(models)


opencode_free = OpenCodeFreeProfile(
    name="opencode-free",
    aliases=("oc-free",),
    display_name="OpenCode Free",
    description="Unofficial OpenCode free-model compatibility with Hermes-local tools",
    api_mode="chat_completions",
    auth_type="external_process",
    env_vars=(),
    base_url=LOGICAL_BASE_URL,
    # Hermes 0.21 routes credential-free custom clients through this seam. The
    # executable is Hermes' own Python; the provider never spawns it.
    process_command=sys.executable,
    process_command_env_vars=(),
    fallback_models=FALLBACK_MODELS,
    default_aux_model=DEFAULT_MODEL,
    model_capabilities={model: {"supports_tools": True} for model in FALLBACK_MODELS},
)

register_provider(opencode_free)
