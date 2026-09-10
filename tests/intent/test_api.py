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

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from llamafactory.intent.models import IntentError
from llamafactory.intent.providers import DEFAULT_CONFIG, APIProvider, ProviderError, read_config


@pytest.fixture
def api_server():
    calls, responses = [], []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            calls.append((self.path, dict(self.headers), None))
            code, body = responses.pop(0)
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            if code == 302:
                self.send_header("Location", "/redirected")
            self.end_headers()
            self.wfile.write(json.dumps(body).encode())

        def do_POST(self):
            calls.append(
                (self.path, dict(self.headers), json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            )
            code, body = responses.pop(0)
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            if code == 302:
                self.send_header("Location", "/redirected")
            self.end_headers()
            self.wfile.write(json.dumps(body).encode())

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield {**DEFAULT_CONFIG, "base_url": f"http://127.0.0.1:{server.server_port}/v1"}, calls, responses
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)


def test_openai_compatible_request(api_server):
    config, calls, responses = api_server
    responses.append(
        (200, {"choices": [{"message": {"content": '{"samples":["退款到账了吗？"]}'}, "finish_reason": "stop"}]})
    )
    provider = APIProvider(config, api_key="TEST_ONLY")
    result = provider.generate([{"role": "user", "content": "Generate JSON samples"}])
    assert json.loads(result)["samples"] == ["退款到账了吗？"]
    path, headers, body = calls[0]
    assert path == "/v1/chat/completions"
    assert headers["Authorization"] == "Bearer TEST_ONLY"
    assert body["response_format"] == {"type": "json_object"}
    assert body["thinking"] == {"type": "disabled"}
    assert "TEST_ONLY" not in json.dumps(provider.descriptor)


@pytest.mark.parametrize(
    "code,retryable",
    [(400, False), (401, False), (402, False), (403, False), (404, False), (429, True), (500, True), (503, True)],
)
def test_http_failures_never_echo_credentials(api_server, code, retryable):
    config, calls, responses = api_server
    responses.append((code, {"error": "request included a secret: TEST_ONLY"}))
    with pytest.raises(ProviderError) as captured:
        APIProvider(config, api_key="TEST_ONLY").generate([{"role": "user", "content": "json"}])
    assert captured.value.retryable == retryable
    assert "TEST_ONLY" not in str(captured.value)
    assert len(calls) == 1


def test_redirect_does_not_forward_auth(api_server):
    config, calls, responses = api_server
    responses.append((302, {}))
    with pytest.raises(ProviderError):
        APIProvider(config, api_key="TEST_ONLY").generate([{"role": "user", "content": "json"}])
    assert len(calls) == 1


def test_truncated_response_is_not_accepted(api_server):
    config, calls, responses = api_server
    responses.append((200, {"choices": [{"message": {"content": '{"samples":['}, "finish_reason": "length"}]}))
    with pytest.raises(ProviderError, match="截断") as captured:
        APIProvider(config, api_key="TEST_ONLY").generate([{"role": "user", "content": "json"}])
    assert captured.value.retryable


def test_configuration_replacement_and_secret_rejection(tmp_path):
    path = tmp_path / "generation.json"
    config = {**DEFAULT_CONFIG, "base_url": "https://my-model.example/v1", "model": "my-model", "thinking": None}
    path.write_text(json.dumps(config), encoding="utf-8")
    assert read_config(path)["model"] == "my-model"
    config["api_key"] = "SHOULD_NOT_BE_HERE"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(IntentError, match="未知字段"):
        read_config(path)


@pytest.mark.parametrize(
    "url", ["https://user:secret@example.com", "http://remote.example.com", "https://x.com?key=abc"]
)
def test_config_rejects_credential_urls(tmp_path, url):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({**DEFAULT_CONFIG, "base_url": url}), encoding="utf-8")
    with pytest.raises(IntentError):
        read_config(path)


def test_env_config_precedence_and_endpoint_credential_isolation(tmp_path, monkeypatch):
    from llamafactory.intent import providers

    path = tmp_path / "config.json"
    path.write_text(json.dumps(DEFAULT_CONFIG), encoding="utf-8")
    monkeypatch.setattr(
        providers,
        "local_settings",
        lambda: {
            "WY_INTENT_BASE_URL": "http://192.168.1.20:8000",
            "WY_INTENT_MODEL": "",
            "WY_INTENT_BATCH_SIZE": "3",
            "DEEPSEEK_API_KEY": "old-service-secret",
        },
    )
    config = read_config(path)
    assert config["base_url"] == "http://192.168.1.20:8000/v1"
    assert config["model"] == "auto" and config["batch_size"] == 3
    assert config["thinking"] is None and config["json_mode"] is False
    assert config["api_key_env"] == "WY_INTENT_API_KEY"
    assert providers.configured_key(config) == ""
    monkeypatch.setenv("WY_INTENT_MODEL", "served-qwen-name")
    monkeypatch.setenv("WY_INTENT_JSON_MODE", "true")
    assert read_config(path)["model"] == "served-qwen-name"
    assert read_config(path)["json_mode"] is True


@pytest.mark.parametrize("key,value", [("JSON_MODE", "maybe"), ("BATCH_SIZE", "hello"), ("THINKING", "other")])
def test_invalid_env_values(tmp_path, monkeypatch, key, value):
    monkeypatch.setenv("WY_INTENT_" + key, value)
    with pytest.raises(IntentError):
        read_config()


def test_single_model_discovery_and_unauthenticated_generation(api_server):
    config, calls, responses = api_server
    config.update(model="auto", thinking=None, json_mode=False)
    responses.extend(
        [(200, {"data": [{"id": "my-served-qwen"}]}), (200, {"choices": [{"message": {"content": '{"samples":[]}'}}]})]
    )
    provider = APIProvider(config)
    assert provider.config["model"] == provider.descriptor["model"] == "my-served-qwen"
    assert provider.generate([]) == '{"samples":[]}'
    assert [x[0] for x in calls] == ["/v1/models", "/v1/chat/completions"]
    assert all("Authorization" not in headers for _, headers, _ in calls)
    assert calls[-1][2]["model"] == "my-served-qwen"
    assert "thinking" not in calls[-1][2] and "response_format" not in calls[-1][2]


@pytest.mark.parametrize("data,match", [([], "没有返回"), ([{"id": "a"}, {"id": "b"}], "多个模型")])
def test_model_discovery_requires_unambiguous_model(api_server, data, match):
    config, calls, responses = api_server
    responses.append((200, {"data": data}))
    with pytest.raises(ProviderError, match=match):
        APIProvider({**config, "model": "auto"})
    assert len(calls) == 1


@pytest.mark.parametrize("code", [302, 401, 404, 500])
def test_model_discovery_errors_are_sanitized_and_never_redirect(api_server, code):
    config, calls, responses = api_server
    responses.append((code, {"error": "secret TEST_ONLY"}))
    with pytest.raises(ProviderError) as error:
        APIProvider({**config, "model": "auto"}, api_key="TEST_ONLY")
    assert "TEST_ONLY" not in str(error.value)
    assert len(calls) == 1


def test_explicit_model_skips_discovery(api_server):
    config, calls, responses = api_server
    assert APIProvider({**config, "model": "manual-qwen"}).config["model"] == "manual-qwen"
    assert calls == []


def test_empty_environment_key_overrides_dotenv(monkeypatch):
    from llamafactory.intent import providers

    monkeypatch.setattr(providers, "local_settings", lambda: {"WY_INTENT_API_KEY": "file-secret"})
    monkeypatch.setenv("WY_INTENT_API_KEY", "")
    assert providers.configured_key({"api_key_env": "WY_INTENT_API_KEY"}) == ""


def test_settings_discovery_returns_exact_names_without_key(api_server, tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from llamafactory.intent.api import create_app
    from llamafactory.intent.service import IntentService

    config, calls, responses = api_server
    monkeypatch.setenv("WY_INTENT_BASE_URL", config["base_url"])
    monkeypatch.setenv("WY_INTENT_API_KEY", "TEST_ONLY")
    responses.append((200, {"data": [{"id": "qwen-one"}, {"id": "qwen-two"}]}))
    service = IntentService(tmp_path)
    try:
        with TestClient(create_app(service=service)) as client:
            settings = client.get("/intent-api/settings")
            assert settings.json()["api_key_required"] is False
            assert settings.json()["model"] == "auto"
            result = client.post("/intent-api/settings/discover")
            assert result.json() == {"models": ["qwen-one", "qwen-two"]}
            assert "TEST_ONLY" not in result.text + settings.text
    finally:
        service.lock.close()
