"""Constrained native -> Hermes tool mappings (oc_translate.NATIVE_MAP).

Pure translation checks plus the real Hermes handlers for the mapped calls; no
binary, no network. Hermes state is isolated before any core import.
"""
from __future__ import annotations

import atexit
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

_SCRATCH = Path(os.environ.get("TMPDIR") or tempfile.gettempdir())
os.environ["HERMES_HOME"] = tempfile.mkdtemp(prefix="oc-native-map-home-", dir=str(_SCRATCH))
atexit.register(shutil.rmtree, os.environ["HERMES_HOME"], ignore_errors=True)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import oc_translate as T  # noqa: E402


def _tool(name: str) -> dict:
    return {"type": "function", "function": {
        "name": name, "description": f"Hermes {name}", "parameters": {"type": "object", "properties": {}}}}


ALL_NATIVE = ["read", "write", "glob", "grep", "question", "todowrite", "edit", "patch", "shell",
              "skill", "subagent", "webfetch", "websearch", "experimental_image_generate"]
HERMES = [_tool(name) for name in (
    "read_file", "write_file", "search_files", "clarify", "todo_list", "patch", "terminal",
    "process_manage", "delegate_task", "web_search", "web_extract", "skill_view", "execute_code")]


class CatalogTests(unittest.TestCase):
    def test_replaced_schemas_are_removed_and_the_rest_kept(self):
        plan = T.plan_tools(HERMES, ALL_NATIVE)
        self.assertEqual(plan.native, {"read", "write", "glob", "grep", "question", "todowrite"})
        wires = {row["name"] for row in plan.register}
        for replaced in ("read_file", "write_file", "search_files", "clarify", "todo_list"):
            self.assertNotIn(replaced, wires, f"{replaced} duplicated beside its native descriptor")
        # Genuine Hermes patch survives by ownership under its reversible alias.
        self.assertEqual(plan.host_to_wire["patch"], "hermes_patch")
        for kept in ("terminal", "process_manage", "delegate_task", "web_search", "web_extract",
                     "skill_view", "execute_code"):
            self.assertEqual(plan.host_to_wire[kept], kept)

    def test_native_kept_only_while_target_enabled(self):
        plan = T.plan_tools([_tool("read_file"), _tool("terminal")], ALL_NATIVE)
        self.assertEqual(plan.native, {"read"})
        self.assertEqual(T.translate_call("write", {"path": "a", "content": "b"}, plan=plan),
                         ("write", {"path": "a", "content": "b"}))
        # todowrite absent from the native registry -> todo_list stays a Hermes schema.
        plan = T.plan_tools([_tool("read_file"), _tool("todo_list")], ["read"])
        self.assertEqual(plan.native, {"read"})
        self.assertIn("todo_list", plan.host_to_wire)

    def test_withheld_hermes_name_still_dispatches_as_itself(self):
        plan = T.plan_tools(HERMES, ALL_NATIVE)
        self.assertEqual(T.translate_call("read_file", {"path": "a"}, plan=plan), ("read_file", {"path": "a"}))
        # A native name owned by Hermes (patch) still never runs Hermes patch.
        self.assertEqual(T.translate_call("patch", {"patchText": "x"}, plan=plan)[0], "oc_unsupported_patch")

    def test_shared_target_replaced_only_when_every_native_exists(self):
        plan = T.plan_tools([_tool("read_file"), _tool("search_files")], ["read", "grep"])
        self.assertEqual(plan.native, {"read"})
        self.assertIn("search_files", plan.host_to_wire)

    def test_guidance_names_limits(self):
        text = T.native_guidance(T.plan_tools(HERMES, ALL_NATIVE))
        for needle in ("Hermes current working directory", "hidden:true is refused", "at most 5 questions",
                       "Native edit, patch and shell are not available"):
            self.assertIn(needle, text)


class ArgumentTests(unittest.TestCase):
    def test_read_limit_zero_default_negative_offset_refused(self):
        self.assertEqual(T.translate_call("read", {"path": "a", "limit": 0}), ("read_file", {"path": "a"}))
        with self.assertRaises(T.NativeUnsupported):
            T.translate_call("read", {"path": "a", "offset": -1})

    def test_runtime_paths_refused(self):
        root = str(_SCRATCH / "hermes-oc-native-x")
        for name, arguments in (("read", {"path": root + "/project/AGENTS.md"}),
                                ("write", {"path": root, "content": ""}),
                                ("glob", {"pattern": "*", "path": root + "/project"}),
                                ("grep", {"pattern": "x", "path": root + "/home"})):
            with self.subTest(name=name), self.assertRaises(T.NativeUnsupported):
                T.translate_call(name, arguments, forbidden_roots=[root])
        self.assertEqual(T.translate_call("read", {"path": "rel.txt"}, forbidden_roots=[root])[1], {"path": "rel.txt"})

    def test_glob(self):
        self.assertEqual(T.translate_call("glob", {"pattern": "*.py", "path": "src", "limit": 7}),
                         ("search_files", {"pattern": "*.py", "target": "files", "path": "src", "limit": 7}))
        self.assertEqual(T.translate_call("glob", {"pattern": "*", "hidden": False})[1]["limit"], 100)
        with self.assertRaises(T.NativeUnsupported):
            T.translate_call("glob", {"pattern": "*", "hidden": True})
        with self.assertRaises(T.NativeInvalidArguments):
            T.translate_call("glob", {"pattern": "*", "limit": 0})

    def test_grep(self):
        self.assertEqual(
            T.translate_call("grep", {"pattern": "a.b(", "literal": True, "caseSensitive": False,
                                      "include": "*.txt", "limit": 3}),
            ("search_files", {"pattern": "(?i)a\\.b\\(", "target": "content", "file_glob": "*.txt", "limit": 3}))
        self.assertEqual(T.translate_call("grep", {"pattern": "x", "caseSensitive": True})[1]["pattern"], "x")
        with self.assertRaises(T.NativeInvalidArguments):
            T.translate_call("grep", {"pattern": "x", "context": 2})

    def test_question_folds_header_and_descriptions(self):
        native = {"questions": [{"question": "Which db?", "header": "Storage",
                                 "options": [{"label": "pg", "description": "Postgres"},
                                             {"label": "lite", "description": ""}], "multiple": True}]}
        name, mapped = T.translate_call("question", native)
        self.assertEqual(name, "clarify")
        self.assertEqual(mapped, {"questions": [{"question": "Storage\n\nWhich db?\n\n- pg: Postgres",
                                                 "choices": ["pg", "lite"], "multi_select": True}]})
        call_id = T.encode_call_id("question", native, mapped, "q1")
        self.assertEqual(T.decode_call_id(call_id)[1], native, "header/descriptions not reversible")

    def test_question_limits_refuse_not_truncate(self):
        q = {"question": "q", "header": "h", "options": []}
        with self.assertRaises(T.NativeUnsupported):
            T.translate_call("question", {"questions": [q] * 6})
        options = [{"label": str(i), "description": ""} for i in range(5)]
        with self.assertRaises(T.NativeUnsupported):
            T.translate_call("question", {"questions": [{**q, "options": options}]})

    def test_question_malformed_values_refuse(self):
        for fields in ({"options": False}, {"options": None}, {"options": [] , "multiple": "yes"}):
            with self.subTest(fields=fields), self.assertRaises(T.NativeInvalidArguments):
                T.translate_call("question", {"questions": [{"question": "q", **fields}]})

    def test_native_json_errors_are_not_replayed_as_success(self):
        for wire, body, kind in (
            ("write", {"error": "read first"}, "error"),
            ("read", {"success": False}, "error"),
            ("question", {"outcome": "cancelled", "responses": []}, "error"),
            ("question", {"outcome": "submitted", "responses": []}, "text"),
            ("terminal", {"error": "opaque passthrough"}, "text"),
        ):
            with self.subTest(wire=wire, body=body):
                cid = T.encode_call_id(wire, {}, {}, "result-test")
                result = T.snapshot_from_openai([
                    {"role": "user", "content": "test"},
                    {"role": "assistant", "tool_calls": [{"id": cid, "function": {
                        "name": wire, "arguments": "{}"}}]},
                    {"role": "tool", "tool_call_id": cid, "content": json.dumps(body)},
                ]).messages[-1]["content"][0]["result"]
                self.assertEqual(result, {"type": kind, "value": json.dumps(body)})

    def test_todowrite(self):
        native = {"todos": [{"content": "a", "status": "pending"}, {"content": "b", "status": "in_progress"}]}
        self.assertEqual(T.translate_call("todowrite", native), ("todo_list", {"todos": [
            {"id": "1", "content": "a", "status": "pending"},
            {"id": "2", "content": "b", "status": "in_progress"}], "merge": False}))
        for bad in ({"todos": [{"content": " ", "status": "pending"}]},
                    {"todos": [{"content": "a", "status": "done"}]},
                    {"todos": [{"content": "a", "status": "pending", "priority": "high"}]}):
            with self.subTest(bad=bad), self.assertRaises(T.NativeInvalidArguments):
                T.translate_call("todowrite", bad)
        with self.assertRaises(T.NativeUnsupported):
            T.translate_call("todowrite", {"todos": [{"content": "x" * 4001, "status": "pending"}]})

    def test_write_roundtrip_is_lossless(self):
        native = {"path": "a", "content": ""}
        name, host = T.translate_call("write", native)
        self.assertEqual((name, host), ("write_file", native))
        self.assertEqual(T.encode_call_id("write", native, host, "w1"), "ocn1.write.v.w1")

    def test_replay_restores_native_name_and_arguments(self):
        plan = T.plan_tools(HERMES, ALL_NATIVE)
        native = {"pattern": "a.b", "literal": True}
        name, host = T.translate_call("grep", native, plan=plan)
        call_id = T.encode_call_id("grep", native, host, "g1")
        snapshot = T.snapshot_from_openai([
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(host)}}]},
            {"role": "tool", "tool_call_id": call_id, "content": "{\"error\": \"boom\"}", "is_error": True},
        ], plan=plan)
        call = snapshot.messages[1]["content"][0]
        self.assertEqual((call["name"], call["input"]), ("grep", native))
        self.assertEqual(snapshot.messages[2]["content"][0]["result"], {"type": "error", "value": "{\"error\": \"boom\"}"})


class HermesHandlerTests(unittest.TestCase):
    """The mapped arguments run unchanged through the real Hermes handlers."""

    @classmethod
    def setUpClass(cls):
        try:
            import model_tools  # noqa: F401
            from tools.registry import registry
        except Exception as error:  # noqa: BLE001
            raise unittest.SkipTest(f"Hermes core not importable: {error}")
        cls.registry = registry
        cls.root = Path(tempfile.mkdtemp(prefix="oc-native-map-", dir=str(_SCRATCH)))
        (cls.root / "a.txt").write_text("call a.b( here\nAXB\n", encoding="utf-8")
        (cls.root / ".hid").mkdir()
        (cls.root / ".hid" / "h.txt").write_text("a.b(\n", encoding="utf-8")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root, ignore_errors=True)

    def _run(self, native, arguments, **kw):
        name, mapped = T.translate_call(native, arguments)
        return json.loads(self.registry.dispatch(name, mapped, **kw))

    def test_grep_literal_and_case(self):
        out = self._run("grep", {"pattern": "a.b(", "literal": True, "path": str(self.root)})
        self.assertNotIn("error", out, out)
        self.assertEqual(out["total_count"], 1, out)  # hidden .hid/h.txt is skipped (documented)
        out = self._run("grep", {"pattern": "axb", "caseSensitive": False, "path": str(self.root)})
        self.assertEqual(out["total_count"], 1, out)

    def test_glob_files(self):
        out = self._run("glob", {"pattern": "*.txt", "path": str(self.root)})
        self.assertEqual([Path(p).name for p in out["files"]], ["a.txt"], out)

    def test_clarify_receives_all_choices(self):
        seen = {}

        def callback(questions):
            seen["questions"] = questions
            return {"answers": {q["qid"]: q["choices_offered"][-1] for q in questions}, "outcome": "submitted"}

        from tools import clarify_tool
        name, mapped = T.translate_call("question", {"questions": [{
            "question": "Pick", "header": "H", "options": [{"label": str(i), "description": ""} for i in range(4)]}]})
        out = json.loads(clarify_tool.clarify_tool(mapped["questions"], callback=callback))
        self.assertEqual(seen["questions"][0]["choices_offered"], ["0", "1", "2", "3"])
        self.assertEqual(out["outcome"], "submitted")

    def test_todowrite_uses_the_given_hermes_store(self):
        from tools.todo_tool import TodoStore

        store = TodoStore()
        out = self._run("todowrite", {"todos": [{"content": "a", "status": "pending"}]}, store=store)
        self.assertEqual(out["todos"], [{"id": "1", "content": "a", "status": "pending"}])
        self.assertEqual(store.read(), out["todos"])


if __name__ == "__main__":
    unittest.main()
