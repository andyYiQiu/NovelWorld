"""Pydantic JSON Schema for SDK tools, retaining shared definitions and validation.

Pydantic emits $defs/$ref; AutoGen's default BaseTool.schema expands every use.
Keep that standard representation on the wire, without changing argument types.
"""


def compact_schema(value):
    """Remove schema titles only; defaults, enums and business fields are intact."""
    if isinstance(value, list):
        return [compact_schema(item) for item in value]
    if isinstance(value, dict):
        return {key: (item if key in ('default', 'const', 'enum', 'examples') else compact_schema(item))
                for key, item in value.items() if not (key == 'title' and isinstance(item, str))}
    return value
