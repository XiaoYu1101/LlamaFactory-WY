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

import threading
from contextlib import contextmanager
from functools import wraps

from .models import IntentError


class ResourceGate:
    """One process-wide device reservation, releasable across generator thread changes."""

    def __init__(self):
        self._lock = threading.Lock()
        self.owner = ""

    @contextmanager
    def reserve(self, owner: str):
        if not self._lock.acquire(blocking=False):
            raise IntentError(f"模型资源正由“{self.owner}”使用，请等其结束或停止后重试。")
        self.owner = owner
        try:
            yield
        finally:
            self.owner = ""
            self._lock.release()


DEVICE = ResourceGate()


def exclusive_resource(owner: str):
    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            with DEVICE.reserve(owner):
                yield from function(*args, **kwargs)

        return wrapped

    return decorate
