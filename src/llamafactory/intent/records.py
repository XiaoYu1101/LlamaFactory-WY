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

"""Full instruction/input/output records; scenario IDs never become answers."""

import json
import re

from .models import IntentError, clean_text, dump, fingerprint


FIELDS = ("instruction", "input", "output")


def record_fingerprint(record):
    return fingerprint(dump([record["instruction"], record["input"]]))


def validate_record(value, scenario=None, generated=False):
    if not isinstance(value, dict) or set(value) != set(FIELDS):
        raise IntentError("每条 JSON 必须且只能包含 instruction、input、output 三个字符串字段。")
    result = {key: clean_text(value[key], key, 12000 if key == "instruction" else 4000) for key in FIELDS}
    if scenario:
        if generated and result["instruction"] not in {x["instruction"] for x in scenario["examples"]}:
            raise IntentError("生成记录的 instruction 必须保持参考样例中的指令，不得自行改写。")
        labels = scenario.get("output_labels", [])
        if labels:
            selected = [x.strip() for x in result["output"].split("、")]
            if not all(selected) or len(set(selected)) != len(selected) or not set(selected) <= set(labels):
                raise IntentError("output 必须选用配置的答案标签，多个标签用顿号分隔，不得重复。")
            # Keep answers stable for training without turning combinations into new classes.
            result["output"] = "、".join(x for x in labels if x in selected)
    return result


def parse_examples(value):
    if isinstance(value, str):
        raw = value.lstrip("\ufeff").strip()
        if len(raw.encode("utf-8")) > 256000:
            raise IntentError("参考样例不能超过 256 KB，请选择少量代表性样例。")
        try:
            value = json.loads(raw)
        except ValueError:
            value = []
            for line_number, line in enumerate(raw.splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    value.append(json.loads(line))
                except ValueError:
                    raise IntentError(
                        f"参考样例第 {line_number} 行不是有效 JSON。支持 JSON 数组或每行一个 JSON 对象。"
                    ) from None
    if isinstance(value, dict):
        value = [value]
    if not isinstance(value, list) or not 1 <= len(value) <= 100:
        raise IntentError("每个生成场景需要 1 到 100 条完整 JSON 参考样例。")
    result, seen = [], {}
    for index, row in enumerate(value, 1):
        try:
            record = validate_record(row)
        except IntentError as error:
            raise IntentError(f"第 {index} 条样例：{error}") from None
        key = record_fingerprint(record)
        if key in seen and seen[key] != record["output"]:
            raise IntentError(f"第 {index} 条样例与前面的相同指令、问句具有不同答案，请先修正。")
        if key not in seen:
            result.append(record)
            seen[key] = record["output"]
    if len(dump(result).encode("utf-8")) > 256000:
        raise IntentError("参考样例不能超过 256 KB。")
    return result


def suggest_labels(examples):
    """Offer explicit label lists written in instructions; user can edit/clear them."""
    candidates = []
    for row in examples:
        match = re.search(r"(?:标签|类别)[：:]\s*([^。；\n]+)", row["instruction"])
        if not match:
            return []
        labels = [x.strip() for x in re.split("[、，,]", match[1].rstrip(". "))]
        if not 2 <= len(labels) <= 100 or any(not x or len(x) > 64 for x in labels):
            return []
        candidates.append(labels)
    return candidates[0] if candidates and all(x == candidates[0] for x in candidates) else []
