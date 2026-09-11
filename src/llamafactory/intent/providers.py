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

import ipaddress
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from contextlib import nullcontext
from pathlib import Path

from .models import IntentError, integer
from .resources import DEVICE


DEFAULT_CONFIG = {
    "provider": "openai_compatible",
    "base_url": "https://api.deepseek.com",
    "model": "deepseek-v4-flash",
    "api_key_env": "DEEPSEEK_API_KEY",
    "timeout_seconds": 60,
    "max_tokens": 2048,
    "temperature": 1.0,
    "json_mode": True,
    "thinking": "disabled",
    "batch_size": 10,
    "seed_count": 3,
}


class ProviderError(IntentError):
    def __init__(self, message: str, retryable: bool = False):
        super().__init__(message)
        self.retryable = retryable


def local_settings(path: str | Path = ".env") -> dict:
    values = {}
    path = Path(path)
    if path.is_file():
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            if line.strip() and not line.lstrip().startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                values[key.strip()] = value.strip().strip("\"'")
    return values


def private_endpoint(url):
    host = urllib.parse.urlsplit(url).hostname
    if host == "localhost":
        return True
    try:
        address = ipaddress.ip_address(host or "")
        return address.is_loopback or address.is_private
    except ValueError:
        return False


def configured_key(config):
    name = config["api_key_env"]
    return os.environ[name] if name in os.environ else local_settings().get(name, "")


def read_config(path: str | Path | None = None) -> dict:
    if path is None:
        path = os.getenv("LF_INTENT_CONFIG", "config/intent_generation.json")
    path = Path(path).expanduser()
    config = dict(DEFAULT_CONFIG)
    if path.is_file():
        try:
            content = json.loads(path.read_text(encoding="utf-8-sig"))
        except (ValueError, OSError):
            raise IntentError("生成配置无法读取，请检查 JSON 文件。") from None
        if not isinstance(content, dict) or set(content) - set(DEFAULT_CONFIG):
            raise IntentError("生成配置包含未知字段；密钥请通过 api_key_env 指定的环境变量配置。")
        config.update(content)
    elif str(path) != "config/intent_generation.json":
        raise IntentError("指定的生成配置文件不存在。")
    values = {**local_settings(), **os.environ}
    overrides = {
        key: values[f"LF_INTENT_{key.upper()}"] for key in DEFAULT_CONFIG if f"LF_INTENT_{key.upper()}" in values
    }
    # Switching endpoints must never implicitly send the previous service's credential.
    if "base_url" in overrides or "LF_INTENT_API_KEY" in values:
        config["api_key_env"] = "LF_INTENT_API_KEY"
    if "base_url" in overrides:
        config.update(model="auto", json_mode=False, thinking=None)
    for key, value in overrides.items():
        if key == "json_mode":
            if value.lower() not in {"true", "false", "1", "0"}:
                raise IntentError("LF_INTENT_JSON_MODE 需要为 true 或 false。")
            value = value.lower() in {"true", "1"}
        elif key == "thinking" and value.lower() in {"", "none", "null"}:
            value = None
        config[key] = value
    if isinstance(config["base_url"], str):
        config["base_url"] = config["base_url"].strip().rstrip("/")
    if isinstance(config["model"], str):
        config["model"] = config["model"].strip() or "auto"
    if config["provider"] not in {"openai_compatible", "local"}:
        raise IntentError("provider 必须为 openai_compatible 或 local。")
    for field, minimum, maximum in (
        ("batch_size", 1, 10),
        ("seed_count", 1, 3),
        ("timeout_seconds", 5, 180),
        ("max_tokens", 128, 8192),
    ):
        config[field] = integer(config[field], field, minimum, maximum)
    if not isinstance(config["model"], str):
        raise IntentError("生成配置需要填写 model。")
    try:
        temperature = float(config["temperature"])
    except (ValueError, TypeError):
        raise IntentError("temperature 必须在 0 到 2 之间。") from None
    if not 0 <= temperature <= 2:
        raise IntentError("temperature 必须在 0 到 2 之间。")
    config["temperature"] = temperature
    if config["thinking"] not in {"disabled", "enabled", None} or not isinstance(config["json_mode"], bool):
        raise IntentError("thinking 可选 disabled、enabled 或 null；json_mode 必须为布尔值。")
    if not isinstance(config["api_key_env"], str) or not config["api_key_env"].isidentifier():
        raise IntentError("api_key_env 必须是合法的环境变量名。")
    parsed = urllib.parse.urlsplit(config["base_url"])
    if parsed.scheme != "https" and not (parsed.scheme == "http" and private_endpoint(config["base_url"])):
        raise IntentError("API 地址需要使用 HTTPS；本机或局域网 IP 允许 HTTP。")
    if not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise IntentError("API 地址不能携带账号、密钥、查询参数或片段。")
    if not parsed.path and parsed.hostname != "api.deepseek.com":
        config["base_url"] += "/v1"
    return config


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Authorization must never be forwarded to a redirected host.
        return None


class APIProvider:
    def __init__(self, config: dict, api_key: str | None = None, resolve_model=True):
        self.config = dict(config)
        self._api_key = configured_key(config) if api_key is None else api_key
        if not self._api_key and not private_endpoint(config["base_url"]):
            raise ProviderError(f"尚未配置 API 密钥，请在本机 .env 或环境变量中设置 {config['api_key_env']}。")
        self.opener = urllib.request.build_opener(NoRedirect())
        if resolve_model and self.config["model"].lower() in {"", "auto"}:
            models = self.models()
            if len(models) != 1:
                raise ProviderError(
                    "服务提供多个模型，请在 .env 的 LF_INTENT_MODEL 中填写一个完整名称：" + "、".join(models[:20])
                )
            self.config["model"] = models[0]
        self.descriptor = {"kind": "openai_compatible", **{k: v for k, v in self.config.items() if k != "api_key_env"}}

    def headers(self):
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def models(self):
        request = urllib.request.Request(self.config["base_url"].rstrip("/") + "/models", headers=self.headers())
        try:
            with self.opener.open(request, timeout=min(10, self.config["timeout_seconds"])) as response:
                raw = response.read(1024 * 1024 + 1)
            if len(raw) > 1024 * 1024:
                raise ProviderError("模型列表过大，请手动填写 LF_INTENT_MODEL。")
            data = json.loads(raw)["data"]
            if not isinstance(data, list):
                raise ValueError("invalid model list")
            models = list(
                dict.fromkeys(
                    x["id"]
                    for x in data
                    if isinstance(x, dict) and isinstance(x.get("id"), str) and x["id"].strip() and len(x["id"]) <= 512
                )
            )
            if not models:
                raise ProviderError("服务没有返回可用模型，请先加载模型，或手动填写 LF_INTENT_MODEL。")
            return models
        except ProviderError:
            raise
        except urllib.error.HTTPError as error:
            raise ProviderError(
                f"获取模型列表失败（HTTP {error.code}），请检查服务地址与密钥；不支持 /models 的服务需手动填写 LF_INTENT_MODEL。"
            ) from None
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
            raise ProviderError("无法连接模型列表接口，请检查服务地址及模型服务是否已启动。", retryable=True) from None
        except (ValueError, KeyError, TypeError):
            raise ProviderError("模型列表不是兼容格式，请手动填写 LF_INTENT_MODEL。") from None

    def session(self):
        return nullcontext()

    def generate(self, messages: list[dict]) -> str:
        body = {
            "model": self.config["model"],
            "messages": messages,
            "stream": False,
            "max_tokens": self.config["max_tokens"],
            "temperature": self.config["temperature"],
        }
        if self.config["json_mode"]:
            body["response_format"] = {"type": "json_object"}
        if self.config["thinking"] is not None:
            body["thinking"] = {"type": self.config["thinking"]}
        url = self.config["base_url"].rstrip("/") + "/chat/completions"
        request = urllib.request.Request(
            url,
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers=self.headers(),
            method="POST",
        )
        try:
            with self.opener.open(request, timeout=self.config["timeout_seconds"]) as response:
                raw = response.read(1024 * 1024 + 1)
            if len(raw) > 1024 * 1024:
                raise ProviderError("API 返回内容过大，请减少输出长度。")
            result = json.loads(raw)
            choice = result["choices"][0]
            if choice.get("finish_reason") == "length":
                raise ProviderError("模型输出被截断，请增大 max_tokens 或减小 batch_size。", retryable=True)
            content = choice["message"]["content"]
            if not isinstance(content, str) or not content.strip():
                raise ProviderError("模型没有返回文本。", retryable=True)
            return content
        except ProviderError:
            raise
        except urllib.error.HTTPError as error:
            messages = {
                400: "API 参数被拒绝，请检查模型名称、json_mode 和 thinking 配置。",
                401: "API 认证失败，请检查本机配置的密钥。",
                402: "API 余额不足。",
                403: "API 访问被拒绝，请检查密钥权限。",
                404: "API 地址或模型不存在。",
                429: "API 请求限流，请稍后继续。",
            }
            raise ProviderError(
                messages.get(error.code, f"API 请求失败（HTTP {error.code}）。"),
                retryable=error.code in {408, 429} or error.code >= 500,
            ) from None
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
            raise ProviderError("API 连接失败或超时，请检查网络后继续任务。", retryable=True) from None
        except (ValueError, KeyError, IndexError, TypeError):
            raise ProviderError("API 返回结构不符合对话接口格式。", retryable=True) from None


class LocalProvider:
    def __init__(self, chatter, config: dict):
        if not chatter.loaded:
            raise ProviderError("请先在 Chat 页面加载用于生成数据的模型。")
        self.chatter, self.config, self.engine = chatter, config, chatter.engine
        details = getattr(chatter, "loaded_config", {}) or {}
        if details.get("model_name_or_path") != config["model"]:
            raise ProviderError("Chat 已加载模型与生成配置的 model 不一致，请使用同一模型路径。")
        self.descriptor = {
            "kind": "local",
            "model": details.get("model_name_or_path", config["model"]),
            "template": details.get("template"),
            "adapter": details.get("adapter_name_or_path"),
            **{k: config[k] for k in ("max_tokens", "temperature", "batch_size", "seed_count")},
        }

    def session(self):
        return DEVICE.reserve("意图样本生成")

    def generate(self, messages: list[dict]) -> str:
        if self.chatter.engine is not self.engine:
            raise ProviderError("生成模型已经切换，请重新启动生成任务。")
        tokenizer = self.engine.tokenizer
        # UTF-8 bound plus tokenizer count; reserve space for the model template and output.
        text = "\n".join(x["content"] for x in messages)
        limit = min(getattr(tokenizer, "model_max_length", 32768), 32768)
        if len(tokenizer.encode(text)) + self.config["max_tokens"] + 512 > limit:
            raise ProviderError("当前模型上下文不足，请精简类别说明、样例或降低输出长度。")
        replies = self.chatter.chat(
            [x for x in messages if x["role"] != "system"],
            system=messages[0]["content"],
            max_new_tokens=self.config["max_tokens"],
            temperature=self.config["temperature"],
        )
        return replies[0].response_text
