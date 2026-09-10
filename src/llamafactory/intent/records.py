# Copyright 2026 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Preserve arbitrary JSON record fields, schema, and generation context."""

import copy
import json
import math
import re

from .models import IntentError, dump, fingerprint


JSON_FORMATS = {"alpaca", "json"}
ANSWER_KEYS = ("output", "answer", "response", "completion")


def is_json_scenario(scenario):
    return scenario.get("format") in JSON_FORMATS


def strict_loads(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise IntentError(f"JSON 包含重复字段 {key}，请修正后导入。")
            result[key] = value
        return result

    def invalid_number(value):
        raise IntentError("JSON 不能包含 NaN 或 Infinity。")

    try:
        return json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid_number)
    except RecursionError:
        raise IntentError("JSON 嵌套过深。") from None


def json_schema(value, depth=0):
    if depth > 12:
        raise IntentError("JSON 嵌套不能超过 12 层。")
    if isinstance(value, dict):
        if any(not isinstance(key, str) or not key or len(key) > 128 for key in value):
            raise IntentError("JSON 字段名必须是 1 到 128 字符的字符串。")
        return {
            "type": "object",
            "properties": {k: json_schema(v, depth + 1) for k, v in value.items()},
            "required": list(value),
            "additionalProperties": False,
        }
    if isinstance(value, list):
        variants = []
        for item in value:
            schema = json_schema(item, depth + 1)
            if schema not in variants:
                variants.append(schema)
        return {"type": "array", "items": {"anyOf": variants}}
    if isinstance(value, str):
        if len(value) > 16000 or "\x00" in value:
            raise IntentError("字符串不能超过 16000 字符或包含空字符。")
        return {"type": "string"}
    if value is None:
        return {"type": "null"}
    if isinstance(value, bool):
        return {"type": "boolean"}
    if isinstance(value, int):
        return {"type": "integer"}
    if isinstance(value, float) and math.isfinite(value):
        return {"type": "number"}
    raise IntentError("记录中包含不支持的 JSON 值。")


def check_schema(value, schema, path="$", depth=0):
    if depth > 12:
        raise IntentError("JSON 嵌套不能超过 12 层。")
    kind = schema["type"]
    types = {
        "object": dict,
        "array": list,
        "string": str,
        "boolean": bool,
        "integer": int,
        "number": (int, float),
        "null": type(None),
    }
    if not isinstance(value, types[kind]) or (kind in {"integer", "number"} and isinstance(value, bool)):
        raise IntentError(f"{path} 类型不一致，需要 {kind}。")
    if kind == "object":
        if set(value) != set(schema["properties"]):
            raise IntentError(
                f"{path} 字段不一致；缺少 {sorted(set(schema['properties']) - set(value))}，多余 {sorted(set(value) - set(schema['properties']))}。"
            )
        for key, sub in schema["properties"].items():
            check_schema(value[key], sub, f"{path}.{key}", depth + 1)
    elif kind == "array":
        for i, item in enumerate(value):
            matched = False
            for sub in schema["items"]["anyOf"]:
                try:
                    check_schema(item, sub, f"{path}[{i}]", depth + 1)
                    matched = True
                    break
                except IntentError:
                    continue
            if not matched:
                raise IntentError(f"{path}[{i}] 的字段或类型与参考数组元素不一致。空数组样例只允许生成空数组。")


def protected_context(value, path="$", result=None):
    result = {} if result is None else result
    if isinstance(value, dict):
        for key, item in value.items():
            field = f"{path}.{key}"
            if key in {"system", "instruction", "system_prompt"}:
                result[field] = item
            elif key in {"content", "value"} and (value.get("role") == "system" or value.get("from") == "system"):
                result[field] = item
            else:
                protected_context(item, field, result)
    elif isinstance(value, list):
        for i, item in enumerate(value):
            protected_context(item, f"{path}[{i}]", result)
    return result


def fixed_context(record, scenario=None):
    result = protected_context(record)
    system_field = (scenario or {}).get("training_mapping") or {}
    system_field = system_field.get("system")
    if system_field:
        result[f"$.{system_field}"] = record[system_field]
    return result


def answer_key(record):
    return next((k for k in ANSWER_KEYS if k in record), None)


def record_fingerprint(record):
    # Keep system and metadata in identity; different system prompts are different examples.
    key = answer_key(record)
    if key:
        return fingerprint(dump({k: v for k, v in record.items() if k != key}))
    for field in ("messages", "conversations"):
        messages = record.get(field)
        if isinstance(messages, list):
            context = copy.deepcopy(record)
            context[field] = [
                x
                for x in messages
                if not isinstance(x, dict) or x.get("role", x.get("from")) not in {"assistant", "gpt"}
            ]
            return fingerprint(dump(context))
    return fingerprint(dump(record))


def answer_signature(record):
    key = answer_key(record)
    if key:
        return dump(record[key])
    for field in ("messages", "conversations"):
        if isinstance(record.get(field), list):
            return dump(
                [
                    x
                    for x in record[field]
                    if isinstance(x, dict) and x.get("role", x.get("from")) in {"assistant", "gpt"}
                ]
            )
    return dump(record)


def record_projection(record):
    """Search/display projections only. The complete record is the source of truth."""

    def display(value):
        return value if isinstance(value, str) else dump(value)

    input_key = next((k for k in ("input", "question", "prompt", "messages", "conversations") if k in record), None)
    text = display(record[input_key]) if input_key else dump(record)
    key = answer_key(record)
    return display(record.get("instruction", "")), text, display(record[key]) if key else answer_signature(record)


def stored_record(row, original=False):
    field = "original_record_json" if original else "record_json"
    if row.get(field):
        return json.loads(row[field])
    prefix = "original_" if original else ""
    if row.get(prefix + "instruction"):
        return dict(
            instruction=row[prefix + "instruction"],
            input=row["original" if original else "text"],
            output=row[prefix + "output"],
        )
    return None


def validate_record(value, scenario=None, generated=False):
    if not isinstance(value, dict) or not value:
        raise IntentError("每条样例必须是非空 JSON 对象。字段名和结构由参考样例决定。")
    json_schema(value)  # Validate JSON primitives and nesting, even without a scenario.
    if len(dump(value).encode("utf-8")) > 64000:
        raise IntentError("单条 JSON 记录不能超过 64 KB。")
    result = copy.deepcopy(value)
    if scenario:
        schema = scenario.get("schema") or json_schema(scenario["examples"][0])
        check_schema(result, schema)
        if generated and fixed_context(result, scenario) not in [
            fixed_context(x, scenario) for x in scenario["examples"]
        ]:
            raise IntentError("生成记录的 system、instruction 或系统消息必须保持某条参考样例的原内容。")
        labels = scenario.get("output_labels", [])
        if labels:
            key = answer_key(result)
            if key is None or not isinstance(result[key], str):
                raise IntentError("答案标签约束需要字符串类型的 output、answer、response 或 completion 字段。")
            selected = [x.strip() for x in result[key].split("、")]
            if not all(selected) or len(set(selected)) != len(selected) or not set(selected) <= set(labels):
                raise IntentError("答案必须选用配置的答案标签，多个标签用顿号分隔，不得重复。")
            result[key] = "、".join(x for x in labels if x in selected)
    return result


def parse_examples(value):
    if isinstance(value, str):
        raw = value.lstrip("\ufeff").strip()
        if len(raw.encode("utf-8")) > 256000:
            raise IntentError("参考样例不能超过 256 KB，请选择少量代表性样例。")
        try:
            value = strict_loads(raw)
        except ValueError as error:
            if isinstance(error, IntentError):
                raise
            value = []
            for line_number, line in enumerate(raw.splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    value.append(strict_loads(line))
                except ValueError:
                    raise IntentError(
                        f"参考样例第 {line_number} 行不是有效 JSON。支持 JSON 数组或每行一个 JSON 对象。"
                    ) from None
    if isinstance(value, dict):
        value = [value]
    if not isinstance(value, list) or not 1 <= len(value) <= 100:
        raise IntentError("每个生成场景需要 1 到 100 条完整 JSON 参考样例。")
    result, seen, schema = [], {}, None
    for index, row in enumerate(value, 1):
        try:
            record = validate_record(row)
            if schema is None:
                schema = json_schema(record)
            check_schema(record, schema)
        except IntentError as error:
            raise IntentError(f"第 {index} 条样例：{error}") from None
        key, answer = record_fingerprint(record), answer_signature(record)
        if key in seen and seen[key] != answer:
            raise IntentError(f"第 {index} 条样例与前面的相同上下文具有不同答案，请先修正。")
        if key not in seen:
            result.append(record)
            seen[key] = answer
    if len(dump(result).encode("utf-8")) > 256000:
        raise IntentError("参考样例不能超过 256 KB。")
    return result


def suggest_labels(examples):
    candidates = []
    for row in examples:
        if not isinstance(row.get(answer_key(row)), str):
            return []
        instruction = "\n".join(
            x for x in (row.get("system"), row.get("instruction"), row.get("system_prompt")) if isinstance(x, str)
        )
        match = re.search(r"(?:标签|类别)[：:]\s*([^。；\n]+)", instruction)
        if not match:
            return []
        labels = [x.strip() for x in re.split("[、，,]", match[1].rstrip(". "))]
        if not 2 <= len(labels) <= 100 or any(not x or len(x) > 64 for x in labels):
            return []
        candidates.append(labels)
    return candidates[0] if candidates and all(x == candidates[0] for x in candidates) else []
