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

"""Map known/custom text records to native LlamaFactory dataset columns."""

from .models import IntentError


def text_mapping(record, custom=None):
    if custom is not None:
        if not isinstance(custom, dict) or set(custom) - {"prompt", "query", "response", "system"}:
            raise IntentError("训练字段映射仅支持 prompt、query、response、system。")
        columns = {k: v for k, v in custom.items() if v is not None and v != ""}
        if not columns.get("response") or not (columns.get("prompt") or columns.get("query")):
            raise IntentError("训练字段映射需要问题字段和答案字段。")
        if any(not isinstance(v, str) or not isinstance(record.get(v), str) for v in columns.values()):
            raise IntentError("训练映射必须指向样例中存在的字符串字段。")
        if len(set(columns.values())) != len(columns):
            raise IntentError("同一字段不能重复映射到多个训练角色。")
        return columns
    response = next(
        (k for k in ("output", "answer", "response", "completion") if isinstance(record.get(k), str)), None
    )
    prompt = next((k for k in ("instruction", "question", "prompt", "input") if isinstance(record.get(k), str)), None)
    if not response or not prompt:
        return None
    columns = {"prompt": prompt, "response": response}
    if prompt == "instruction" and isinstance(record.get("input"), str):
        columns["query"] = "input"
    for key in ("system", "system_prompt"):
        if key in record:
            if not isinstance(record[key], str):
                return None
            columns["system"] = key
            break
    return columns


def dataset_mapping(record, custom=None):
    columns = text_mapping(record, custom)
    if columns:
        return {"formatting": "alpaca", "columns": columns}
    if custom is not None:
        return None
    for field, role, content, user, assistant in (
        ("messages", "role", "content", "user", "assistant"),
        ("conversations", "from", "value", "human", "gpt"),
    ):
        messages = record.get(field)
        if not isinstance(messages, list) or not messages:
            continue
        if not all(
            isinstance(x, dict) and isinstance(x.get(content), str) and x.get(role) in {"system", user, assistant}
            for x in messages
        ):
            continue
        roles = [x[role] for x in messages]
        if roles[0] == "system":
            roles = roles[1:]
        if not roles or len(roles) % 2 or roles != [user, assistant] * (len(roles) // 2):
            continue
        columns = {"messages": field}
        system_key = next((key for key in ("system", "system_prompt") if key in record), None)
        if system_key:
            if not isinstance(record[system_key], str):
                continue
            # Native ShareGPT gives the initial system message priority over the root column.
            # Reject conflicting contexts instead of silently dropping one during training.
            if messages[0][role] == "system" and messages[0][content] != record[system_key]:
                continue
            columns["system"] = system_key
        return {
            "formatting": "sharegpt",
            "columns": columns,
            "tags": {
                "role_tag": role,
                "content_tag": content,
                "user_tag": user,
                "assistant_tag": assistant,
                "system_tag": "system",
            },
        }
    return None
