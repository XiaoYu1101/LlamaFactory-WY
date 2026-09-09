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

import atexit
import os
import threading
from pathlib import Path

from .jobs import JobRunner
from .models import IntentError
from .providers import APIProvider, LocalProvider, read_config
from .storage import Store


class WorkspaceLock:
    def __init__(self, root: Path):
        root.mkdir(parents=True, exist_ok=True)
        self.file = (root / "service.lock").open("a+b")
        self.file.seek(0, 2)
        if self.file.tell() == 0:
            self.file.write(b"0")
            self.file.flush()
        self.file.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self.file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.file.close()
            raise IntentError("此意图工作区已由另一个服务使用，请关闭旧服务或使用不同的数据目录。") from None

    def close(self):
        if not self.file.closed:
            self.file.close()


class IntentService:
    def __init__(self, root: str | Path, config_path=None):
        root = Path(root).expanduser().resolve()
        self.lock = WorkspaceLock(root)
        try:
            self.store = Store(root)
            self.store.recover_jobs()
            self.jobs = JobRunner(self.store)
            self.config_path = config_path
        except Exception:
            self.lock.close()
            raise
        atexit.register(self.lock.close)

    def provider(self, chatter=None):
        config = read_config(self.config_path)
        if config["provider"] == "local":
            if chatter is None:
                raise IntentError("本地模型模式需在完整 LlamaFactory WebUI 的 Chat 页面加载模型。")
            return LocalProvider(chatter, config)
        return APIProvider(config)

    def start(self, version_id, chatter=None, label=None, amount=None):
        provider = self.provider(chatter)
        config = self.store.version(version_id)["config"]
        targets = {x["label"]: x["target"] for x in config}
        if label:
            if label not in targets:
                raise IntentError("补生成类别不存在。")
            targets = {label: amount}
        job_id = self.store.create_job(version_id, targets, provider.descriptor)
        self.jobs.start(job_id, provider)
        return job_id


_SERVICES = {}
_SERVICE_LOCK = threading.Lock()


def get_service(root=None, config_path=None):
    root = Path(root or os.getenv("WY_INTENT_HOME", "workspace/intent")).expanduser().resolve()
    with _SERVICE_LOCK:
        if root not in _SERVICES:
            _SERVICES[root] = IntentService(root, config_path)
        elif config_path and _SERVICES[root].config_path != config_path:
            raise IntentError("此工作区已有其他生成配置，请重启服务后切换。")
        return _SERVICES[root]
