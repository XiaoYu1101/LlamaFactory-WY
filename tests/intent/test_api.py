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
