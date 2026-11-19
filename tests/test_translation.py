"""Fast read-only translation/provenance regressions; no binary, no network."""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import os
import oc_translate as T  # noqa: E402

# Byte-for-byte registration contract tests run with slimming off;
# tests/test_slim.py covers the compact wire form.
os.environ.setdefault("OC_FREE_SLIM_SCHEMAS", "0")

READ_FILE = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read a file with line numbers.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "offset": {"type": "integer"},
                "limit": {"type": "integer"},
            },
            "required": ["path"],
        },
    },
}

PATCH = {
    "type": "function",
    "function": {"name": "patch", "description": "Patch a file.", "parameters": {}},
}

TERMINAL = {
    "type": "function",
    "function": {"name": "terminal", "description": "Run a command.",
                 "parameters": {"type": "object", "properties": {}}},
}


class PlanTests(unittest.TestCase):
    def test_every_unreplaced_hermes_schema_is_additive_and_exact(self):
        plan = T.plan_tools([READ_FILE, TERMINAL], ["read"])
        # read_file is replaced by native read: one name per capability.
        self.assertEqual(plan.host_to_wire, {"terminal": "bash"})
        self.assertEqual(plan.native, {"read"})
        self.assertEqual(plan.enabled, {"read_file", "terminal"})
        for row, source in zip(plan.register, (TERMINAL,)):
            self.assertEqual(row["parameters"], source["function"]["parameters"])
            self.assertEqual(row["description"], source["function"]["description"])
            self.assertFalse(row["codemode"])

    def test_colliding_hermes_name_is_reversible(self):
        plan = T.plan_tools([READ_FILE, PATCH], ["read", "patch"])
        self.assertEqual(plan.host_to_wire["patch"], "hermes_patch")
        self.assertEqual(plan.wire_to_host["hermes_patch"], "patch")
        self.assertEqual(T.translate_call("hermes_patch", {"path": "a"}, plan=plan), ("patch", {"path": "a"}))

    def test_exposed_native_read_is_reserved(self):
        native_read = {"type": "function", "function": {
            "name": "read", "description": "native read", "parameters": {}}}
        plan = T.plan_tools([READ_FILE, native_read], ["read"])
        self.assertEqual(plan.host_to_wire["read"], "hermes_read")
        self.assertNotIn("read_file", plan.host_to_wire)

    def test_read_file_unavailable_fails_before_inference(self):
        # Approved contract: never advertise the native read adapter without
        # the Hermes tool that actually runs it — including tool-less requests.
        with self.assertRaises(T.CompatibilityUnavailable):
            T.plan_tools([TERMINAL], ["read"])
        with self.assertRaises(T.CompatibilityUnavailable):
            T.plan_tools([], ["read"])


class MappingTests(unittest.TestCase):
    def test_native_read_mapping(self):
        self.assertEqual(set(T.NATIVE_MAP), {"read", "write", "glob", "grep", "question", "todowrite"})
        self.assertEqual(T.translate_call("read", {"path": "a", "offset": 0, "limit": 3}), (
            "read_file", {"path": "a", "offset": 1, "limit": 3}))
        self.assertEqual(T.translate_call("read", {"path": "a"}), ("read_file", {"path": "a"}))
        # Native offset 0 and 1 both mean the first line.
        self.assertEqual(T.translate_call("read", {"path": "a", "offset": 1})[1]["offset"], 1)

    def test_read_limit_is_capped_and_validated(self):
        self.assertEqual(T.translate_call("read", {"path": "a", "limit": 99_999})[1]["limit"],
                         T.READ_LIMIT_MAX)
        # Native limit 0 means the default.
        self.assertEqual(T.translate_call("read", {"path": "a", "limit": 0}), ("read_file", {"path": "a"}))
        for arguments in ({"path": ""}, {"path": "a", "limit": -1},
                          {"path": "a", "extra": 1}, {"offset": 1}):
            with self.subTest(arguments=arguments), self.assertRaises(T.NativeInvalidArguments):
                T.translate_call("read", arguments)

    def test_negative_offset_is_explicitly_unsupported(self):
        with self.assertRaises(T.NativeUnsupported):
            T.translate_call("read", {"path": "a", "offset": -5})

    def test_other_native_families_have_no_translation(self):
        for name, arguments in (
            ("edit", {"path": "a", "oldString": "x", "newString": "y"}),
            ("shell", {"command": "pwd"}),
            ("patch", {"patchText": "*** Begin Patch"}),
            ("skill", {"id": "x"}),
            ("webfetch", {"url": "https://example.invalid"}),
            ("websearch", {"query": "x"}),
            ("subagent", {}),
            ("experimental_image_generate", {"prompt": "x"}),
            ("execute", {"code": "1"}),
        ):
            with self.subTest(name=name):
                self.assertEqual(T.translate_call(name, arguments), (name, arguments))

    def test_disabled_target_never_runs_a_different_tool(self):
        self.assertEqual(T.translate_call("read", {"path": "a"}, enabled={"terminal"}),
                         ("oc_unsupported_read", {"path": "a"}))
        self.assertEqual(T.translate_call("terminal", {"command": "x"}, enabled={"terminal"}),
                         ("oc_unsupported_terminal", {"command": "x"}))

    def test_registered_hermes_identity_is_forwarded_unchanged(self):
        plan = T.plan_tools([READ_FILE, TERMINAL], ["read"])
        arguments = {"command": "printf hi", "timeout": 5}
        self.assertEqual(T.translate_call("terminal", arguments, plan=plan), ("terminal", arguments))


class ProvenanceTests(unittest.TestCase):
    def test_lossless_call_id_has_no_payload(self):
        call_id = T.encode_call_id(
            "read", {"path": "a", "offset": 2},
            {"path": "a", "offset": 2}, "call_1")
        self.assertEqual(call_id, "ocn1.read.v.call_1")
        self.assertEqual(T.decode_call_id(call_id), ("read", None, "call_1"))

    def test_offset_zero_translation_round_trips_through_call_id(self):
        native = {"path": "a", "offset": 0}
        host = T.translate_call("read", native)[1]
        call_id = T.encode_call_id("read", native, host, "call_zero")
        wire, recovered, original = T.decode_call_id(call_id)
        self.assertEqual((wire, original), ("read", "call_zero"))
        self.assertEqual(recovered, native)
        snapshot = T.snapshot_from_openai([
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": "calling", "tool_calls": [{
                "id": call_id, "type": "function",
                "function": {"name": "read_file", "arguments": json.dumps(host)},
            }]},
            {"role": "tool", "tool_call_id": call_id, "name": "read_file", "content": "ok"},
        ])
        self.assertEqual(snapshot.messages[1]["content"][1]["input"], native)

    def test_dotted_wire_names_round_trip(self):
        native = {"sessionID": "ses_1", "directory": "."}
        call_id = T.encode_call_id("opencode.session_move", native, native, "call_dotted")
        self.assertEqual(T.decode_call_id(call_id), ("opencode.session_move", None, "call_dotted"))
        self.assertEqual(len(call_id.split(".", 3)), 4)

    def test_oversized_compressed_payload_is_refused_before_inflation(self):
        import base64
        import zlib

        bomb = base64.urlsafe_b64encode(zlib.compress(b"0" * (T.MAX_CALL_ARGUMENT_BYTES + 1))).decode().rstrip("=")
        with self.assertRaises(T.SnapshotError):
            T.decode_call_id(f"ocn1.read.p{bomb}.call_bomb")

    def test_mixed_roundtrip_keeps_native_and_direct_names(self):
        plan = T.plan_tools([READ_FILE, TERMINAL], ["read"])
        native_id = T.encode_call_id("read", {"path": "a"},
                                     {"path": "a"}, "call_r")
        direct_id = "plain-foreign-id"
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "calling", "tool_calls": [
                {"id": native_id, "type": "function",
                 "function": {"name": "read_file", "arguments": json.dumps({"path": "a"})}},
                {"id": direct_id, "type": "function",
                 "function": {"name": "terminal", "arguments": json.dumps({"command": "pwd"})}},
            ]},
            {"role": "tool", "tool_call_id": native_id, "name": "read_file", "content": "read"},
            {"role": "tool", "tool_call_id": direct_id, "name": "terminal", "content": "ok"},
        ]
        snapshot = T.snapshot_from_openai(messages, plan=plan)
        calls = [item for item in snapshot.messages[1]["content"] if item.get("type") == "tool-call"]
        self.assertEqual([call["name"] for call in calls], ["read", "bash"])
        self.assertEqual(calls[0]["input"], {"path": "a"})
        self.assertEqual(snapshot.messages[2]["content"][0]["name"], "read")
        self.assertEqual(snapshot.messages[3]["content"][0]["name"], "bash")

    def test_orphan_and_duplicate_history_refuse(self):
        with self.assertRaises(T.SnapshotError):
            T.snapshot_from_openai([{"role": "tool", "tool_call_id": "x", "content": "x"},
                                    {"role": "user", "content": "hi"}])
        with self.assertRaises(T.SnapshotError):
            T.snapshot_from_openai([
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "", "tool_calls": [
                    {"id": "dup", "function": {"name": "read", "arguments": "{}"}}]},
                {"role": "tool", "tool_call_id": "dup", "content": "1"},
                {"role": "assistant", "content": "", "tool_calls": [
                    {"id": "dup", "function": {"name": "read", "arguments": "{}"}}]},
            ])


class NativeMetadataTests(unittest.TestCase):
    def test_carrier_preserves_reasoning_text_and_tool_metadata(self):
        parts = {"parts": [
            {"type": "reasoning", "index": 0, "native": {"reasoningField": "reasoning_content"}},
            {"type": "text", "index": 1, "native": {"textField": "content"}},
            {"type": "tool-call", "index": 2, "id": "call_r", "native": {"providerExecuted": False}},
        ]}
        carrier = T.native_carrier(parts)
        self.assertEqual(carrier["type"], T.NATIVE_CARRIER)
        self.assertNotIn("step", carrier)

        call_id = T.encode_call_id("read", {"path": "a"}, {"path": "a"}, "call_r")
        messages = [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": "calling", "reasoning": "why",
             "reasoning_details": [carrier],
             "tool_calls": [{"id": call_id, "type": "function",
                             "function": {"name": "read_file", "arguments": json.dumps({"path": "a"})}}]},
            {"role": "tool", "tool_call_id": call_id, "name": "read_file", "content": "ok"},
        ]
        snapshot = T.snapshot_from_openai(messages)
        content = snapshot.messages[1]["content"]
        self.assertEqual(content[0]["native"], {"reasoningField": "reasoning_content"})
        self.assertEqual(content[1]["native"], {"textField": "content"})
        self.assertEqual(content[2]["native"], {"providerExecuted": False})

    def test_sparse_metadata_does_not_shift_later_parts(self):
        carrier = T.native_carrier({"parts": [
            {"type": "text", "index": 0, "native": None},
            {"type": "text", "index": 1, "native": {"textField": "content"}},
        ]})
        self.assertEqual(carrier["parts"], [{"type": "text", "index": 1, "native": {"textField": "content"}}])
        snapshot = T.snapshot_from_openai([
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": [{"type": "text", "text": "first"},
                                              {"type": "text", "text": "second"}],
             "reasoning_details": [carrier]},
        ])
        first, second = snapshot.messages[1]["content"]
        self.assertNotIn("native", first)
        self.assertEqual(second["native"], {"textField": "content"})

    def test_absent_metadata_is_not_invented(self):
        self.assertIsNone(T.native_carrier({"parts": [{"type": "text", "native": None}]}))
        self.assertIsNone(T.native_carrier(None))
        snapshot = T.snapshot_from_openai([
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": "hello"},
        ])
        self.assertNotIn("native", snapshot.messages[1]["content"][0])


if __name__ == "__main__":
    unittest.main()
