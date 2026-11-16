"""Out-of-tree model-provider registration; import does not start OpenCode."""
from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

from providers import register_provider
from providers.base import ProviderProfile

_PLUGIN_DIR = str(Path(__file__).parent)
if _PLUGIN_DIR not in sys.path:
    sys.path.insert(0, _PLUGIN_DIR)

import oc_binary
from oc_catalog import describe_models, free_model_ids
from oc_translate import NATIVE_CARRIER

# Logical identity only: no HTTP request is made to it. Must match
# hermes_oc_client.LOGICAL_BASE_URL (RUNTIME_API.md), or core's generic catalog
# probe would dial a real host instead of failing closed.
LOGICAL_BASE_URL = "opencode-native://local"


def _auth_handler(action, args):
    """``hermes auth add opencode-free``: validate and save the binary path.

    ``status``/``refresh``/``logout`` return False so Hermes' generic
    external-process behavior stays in charge. No credential-pool entry and no
    fake API key are created.
    """
    if action != "add":
        return False
    saved = oc_binary.ensure()
    if saved is None:
        raise SystemExit(
            "OpenCode is not configured. Run `hermes auth add opencode-free` in a terminal "
            f"and provide the absolute path to an `opencode` >= {'.'.join(str(p) for p in oc_binary.MIN_VERSION)} binary."
        )
    if oc_binary.configured_path() == saved:
        print(f"OpenCode binary configured: {saved}")
    else:
        print(f"Using the validated OpenCode on the Hermes process PATH: {saved}")
    return True


def _model_setup_flow(config, current_model, args):
    """``hermes model`` step: onboard the binary, then pick a free model."""
    from hermes_cli.model_setup_flows import _pick_model_or_prompt, _plugin_flow_live_rows
    from hermes_cli.model_setup_flows_common import _finish_model
    from hermes_cli.models import merge_profile_catalog

    del config, args
    saved = oc_binary.ensure()
    if saved is None:
        return None
    opencode_free.invalidate_catalog_cache()
    live, notes = _plugin_flow_live_rows(opencode_free, "", LOGICAL_BASE_URL)
    model_list = merge_profile_catalog("opencode-free", opencode_free, live) or []
    selected = _pick_model_or_prompt(
        model_list,
        "Model name: ",
        current_model=current_model,
        confirm_provider="opencode-free",
        confirm_base_url=LOGICAL_BASE_URL,
        confirm_api_key="",
        notes=notes,
    )
    _finish_model(
        selected,
        "opencode-free",
        f"Default model set to: {selected} (via {opencode_free.display_name})",
        base_url=LOGICAL_BASE_URL,
        api_mode="chat_completions",
    )
    return None


def _launch_command() -> str:
    """Cheap, profile-scoped command for core's external-process eligibility gate.

    Core resolves this before calling ``create_client`` and hands the result to
    the client as a validated hint. Precedence matches the runtime resolver:
    usable PATH, then the active profile setting, then the bare command. No
    subprocess and no env override, so it is safe to evaluate per call.
    """
    try:
        return shutil.which("opencode") or oc_binary.configured_path() or "opencode"
    except Exception:  # noqa: BLE001 - registration must never fail
        return "opencode"


class OpenCodeFreeProfile(ProviderProfile):
    # Dynamic so external config edits and profile switches are visible to
    # core's credential gate in the same process. The dataclass init assigns
    # this field; the setter keeps that assignment from replacing the property.
    @property
    def process_command(self) -> str:
        return _launch_command()

    @process_command.setter
    def process_command(self, _seed) -> None:
        self.__dict__["_process_command_seed"] = _seed

    def create_client(self, **kwargs):
        from hermes_oc_client import OpenCodeClient
        return OpenCodeClient(**kwargs)

    def build_api_kwargs_extras(
        self, *, reasoning_config=None, **context
    ) -> tuple[dict, dict]:
        """Carry the logical conversation scope as provider-only top-level data.

        ``cache_scope_id`` is the rotation-stable compression lineage; plain
        ``session_id`` rotates on compaction. No id (auxiliary/stateless calls)
        emits nothing, so a generic fallback client can never receive the
        marker as an unknown wire field.
        """
        del reasoning_config
        session_id = context.get("session_id")
        cache_scope_id = context.get("cache_scope_id")
        if not isinstance(session_id, str):
            session_id = ""
        if not isinstance(cache_scope_id, str):
            cache_scope_id = ""
        if not session_id and not cache_scope_id:
            return {}, {}
        return {}, {
            "hermes_session_context": {
                # Both raw ids are required: `sub_` delegate siblings share a
                # cache scope but have distinct physical sessions, while
                # compression rotates the physical id and keeps the scope.
                "version": 2,
                "session_id": session_id,
                "cache_scope_id": cache_scope_id,
            }
        }


    def _catalog_key(self):
        from hermes_constants import get_hermes_home
        return (str(get_hermes_home()), oc_binary.location_hint())

    def _catalog(self, *, refresh=False, timeout=15.0):
        key = self._catalog_key()
        cache = getattr(self, "_native_catalog_cache", {})
        entry = cache.get(key)
        if not refresh and entry is not None and time.monotonic() - entry[0] < 60:
            return entry[1]
        client = None
        try:
            client = self.create_client()
            rows = describe_models(client.list_model_details(timeout=timeout))
        except Exception:
            # Discovery failure differs from a valid empty catalog. No stale price fallback.
            cache.pop(key, None)
            self._native_catalog_cache = cache
            return None
        finally:
            if client is not None:
                try:
                    client.close()
                except Exception:
                    # Discovery hooks must not break Hermes startup on teardown failure.
                    import logging
                    logging.getLogger(__name__).warning("OpenCode discovery cleanup failed", exc_info=True)
        cache[key] = (time.monotonic(), rows)
        self._native_catalog_cache = cache
        return rows

    def fetch_models(self, *, api_key=None, base_url=None, timeout=15.0):
        rows = self._catalog(timeout=timeout)
        return None if rows is None else free_model_ids(rows)

    def picker_models(self, force_refresh=False):
        rows = self._catalog(refresh=force_refresh)
        return None if rows is None else free_model_ids(rows)

    def probe_models(self, *, base_url, api_key=None, timeout=5.0):
        return self.fetch_models(timeout=timeout), LOGICAL_BASE_URL, LOGICAL_BASE_URL

    def invalidate_catalog_cache(self):
        getattr(self, "_native_catalog_cache", {}).pop(self._catalog_key(), None)

    def resolve_aux_model(self, *, vision=False):
        # Auxiliary calls send temperature/max_tokens, which this transport
        # cannot map yet (generation-control parity is deferred). Never
        # advertise a cheap model whose controls would be rejected: return ""
        # so core keeps the caller's model/provider choice and any aux attempt
        # fails loudly instead of silently ignoring controls.
        return ""

opencode_free = OpenCodeFreeProfile(
    name="opencode-free",
    aliases=("oc-free",),
    display_name="OpenCode Free (native)",
    description="Official OpenCode binary and companion; Hermes owns tool effects and each request",
    api_mode="chat_completions",
    auth_type="external_process",
    auth_handler=_auth_handler,
    model_setup_flow=_model_setup_flow,
    env_vars=(),
    base_url=LOGICAL_BASE_URL,
    process_command=_launch_command(),
    fallback_models=(),
    default_aux_model="",
    # Opaque per-part native metadata survives canonical history only for this
    # provider's declared carrier type.
    native_reasoning_details_type=NATIVE_CARRIER,
)
opencode_free.picker_catalog = opencode_free.picker_models
register_provider(opencode_free)
