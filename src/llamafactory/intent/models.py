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

import hashlib
import json
import math
import re
import unicodedata
import uuid
from datetime import UTC, datetime


class IntentError(ValueError):
    """A business error that can be shown directly in the user interface."""


def uid() -> str:
    return uuid.uuid4().hex


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def dump(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def fingerprint(text: str) -> str:
    normalized = " ".join(unicodedata.normalize("NFKC", text).split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def clean_text(value, field: str, maximum: int = 1000) -> str:
    if not isinstance(value, str) or not value.strip():
        raise IntentError(f"{field}不能为空。")
    value = value.strip()
    if len(value) > maximum or "\x00" in value:
        raise IntentError(f"{field}不能超过 {maximum} 个字符，也不能包含空字符。")
    return value


def integer(value, field: str, minimum: int = 1, maximum: int = 100000) -> int:
    try:
        number = float(value)
        if isinstance(value, bool) or not math.isfinite(number) or number != int(number):
            raise ValueError
        result = int(number)
    except (TypeError, ValueError, OverflowError):
        raise IntentError(f"{field}必须是整数。") from None
    if not minimum <= result <= maximum:
        raise IntentError(f"{field}必须在 {minimum} 到 {maximum} 之间。")
    return result


def validate_intents(rows: list[dict]) -> list[dict]:
    from .records import parse_examples, validate_record

    if not isinstance(rows, list) or not 1 <= len(rows) <= 100:
        raise IntentError("请配置 1 到 100 个意图类别。")
    result, labels = [], set()
    for row in rows:
        if not isinstance(row, dict):
            raise IntentError("每个类别必须包含标识、名称、说明和样例。")
        label = clean_text(row.get("label"), "类别标识", 64)
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", label) or label.casefold() in labels:
            raise IntentError("类别标识需以英文字母开头，使用字母、数字、下划线或短横线，且不能重复。")
        labels.add(label.casefold())
        if row.get("format") == "alpaca":
            examples = parse_examples(row.get("examples"))
            output_labels = row.get("output_labels", [])
            if not isinstance(output_labels, list) or len(output_labels) > 100:
                raise IntentError("答案标签必须是最多 100 项的列表。")
            output_labels = [clean_text(x, "答案标签", 64) for x in output_labels]
            if len(set(output_labels)) != len(output_labels) or any("、" in x for x in output_labels):
                raise IntentError("答案标签不能重复或包含顿号。")
            scenario = dict(
                label=label,
                name=clean_text(row.get("name"), "场景名称", 100),
                description=clean_text(row.get("description"), "生成要求"),
                examples=examples,
                target=integer(row.get("target", 1000), "生成数量"),
                format="alpaca",
                output_labels=output_labels,
            )
            scenario["examples"] = [validate_record(x, scenario) for x in examples]
            result.append(scenario)
            continue
        if row.get("format") not in {None, "text"}:
            raise IntentError("不支持的样例格式。")
        seeds = row.get("examples", [])
        if isinstance(seeds, str):
            seeds = [line.strip() for line in seeds.splitlines() if line.strip()]
        if not isinstance(seeds, list) or not 1 <= len(seeds) <= 100:
            raise IntentError(f"{label} 需要 1 到 100 条样例。")
        examples, seen = [], set()
        for seed in seeds:
            seed = clean_text(seed, "样例")
            key = fingerprint(seed)
            if key not in seen:
                examples.append(seed)
                seen.add(key)
        result.append(
            dict(
                label=label,
                name=clean_text(row.get("name"), "类别名称", 100),
                description=clean_text(row.get("description"), "类别说明"),
                examples=examples,
                target=integer(row.get("target", 100), "生成数量"),
            )
        )
    if sum(row["target"] for row in result) > 100000:
        raise IntentError("一次配置的生成总量不能超过 100000 条。")
    # The same prompt is used at training and inference time; never silently omit labels.
    if len(classification_prompt([x for x in result if x.get("format") != "alpaca"]).encode("utf-8")) > 12000:
        raise IntentError("类别定义总长度过大，请精简说明或拆分项目。")
    return result


def classification_prompt(intents: list[dict]) -> str:
    definitions = "\n".join(f"{x['label']}（{x['name']}）：{x['description']}" for x in intents)
    return f"根据以下类别定义判断用户文本的意图。只输出一个类别标识，不输出解释。\n{definitions}"
