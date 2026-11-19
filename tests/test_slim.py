"""Slimming guarantees: names survive, shape survives, output is deterministic.

Run: python3 tests/test_slim.py  (assert-based, no fixtures)
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import oc_translate as t


def test_property_names_and_types_survive():
    schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "The file path.", "minLength": 1},
            "limit": {"type": "integer", "minimum": 1, "default": 5},
        },
        "required": ["path"],
        "additionalProperties": False,
    }
    out = t._slim_schema(schema)
    assert sorted(out["properties"]) == ["limit", "path"], out
    assert out["properties"]["path"] == {"type": "string"}, out
    assert out["properties"]["limit"] == {"type": "integer"}, out
    assert out["required"] == ["path"]
    assert "additionalProperties" not in out and "description" not in out


def test_array_items_survive_recursively():
    schema = {
        "type": "object",
        "properties": {
            "todos": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "description": "Task text"},
                        "status": {"type": "string", "enum": ["pending", "done"]},
                    },
                    "required": ["content", "status"],
                },
            }
        },
        "required": ["todos"],
    }
    out = t._slim_schema(schema)
    items = out["properties"]["todos"]["items"]
    assert items["properties"]["content"] == {"type": "string"}, items
    assert items["properties"]["status"]["enum"] == ["pending", "done"], items
    assert items["required"] == ["content", "status"], items


def test_anyof_collapses_to_first_non_null():
    schema = {"properties": {"notify": {"anyOf": [
        {"type": "null"},
        {"type": "boolean", "description": "Enable"},
        {"type": "array", "items": {"type": "string"}},
    ]}}}
    out = t._slim_schema(schema)
    assert out["properties"]["notify"] == {"type": "boolean"}, out


def test_description_first_sentence_capped():
    assert t._slim_description("Read a file. Supports offsets and more. Extra.") == "Read a file."
    long = "x" * 500
    assert len(t._slim_description(long)) == 300


def test_deterministic():
    schema = {"type": "object", "properties": {"a": {"type": "string", "default": "x"}}}
    assert json.dumps(t._slim_schema(schema), sort_keys=True) == json.dumps(
        t._slim_schema(json.loads(json.dumps(schema))), sort_keys=True)


def test_registration_toggle():
    os.environ[t.SLIM_ENV] = "0"
    fat = t._slim_registration("read_file", "Read a file. More prose.", {"type": "object", "properties": {}})
    assert fat["description"] == "Read a file. More prose."
    os.environ[t.SLIM_ENV] = "1"
    slim = t._slim_registration("read_file", "Read a file. More prose.", {"type": "object", "properties": {}})
    assert slim["description"] == "Read a file."
    os.environ.pop(t.SLIM_ENV, None)


def test_real_catalog_shrinks_and_names_survive():
    sample = os.environ.get("OC_SLIM_SAMPLE", "~/.hermes/cache/scratch/yuqi_tools.json")
    try:
        catalog = json.load(open(os.path.expanduser(sample)))
    except FileNotFoundError:
        print("skip: no captured catalog")
        return
    before = json.dumps(catalog)
    slimmed = [t._slim_registration(c["function"]["name"], c["function"]["description"],
                                    c["function"]["parameters"]) for c in catalog]
    after = json.dumps(slimmed)
    assert len(after) < len(before) * 0.6, (len(before), len(after))
    for slim_tool, orig in zip(slimmed, catalog):
        orig_props = (orig["function"].get("parameters") or {}).get("properties") or {}
        slim_props = (slim_tool.get("parameters") or {}).get("properties") or {}
        assert set(slim_props) <= set(orig_props), (slim_tool["name"], set(slim_props) - set(orig_props))
    print(f"catalog slimmed: {len(before)} -> {len(after)} bytes over {len(catalog)} tools")


def test_wire_rename_flagged_names():
    assert t.WIRE_RENAMES.get("terminal") == "bash"
    hermes = [{"type": "function", "function": {"name": "terminal", "description": "Run a command.",
                "parameters": {"type": "object", "properties": {"command": {"type": "string"}},
                               "required": ["command"]}}},
              {"type": "function", "function": {"name": "read_file", "description": "Read a file.",
                "parameters": {"type": "object", "properties": {"path": {"type": "string"}},
                               "required": ["path"]}}}]
    plan = t.plan_tools(hermes, ["read"])
    names = [r["name"] for r in plan.register]
    assert "bash" in names and "terminal" not in names, names
    # replay still resolves the wire name back to the Hermes tool
    assert plan.wire_to_host["bash"] == "terminal"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok:", name)
    print("ALL PASS")
