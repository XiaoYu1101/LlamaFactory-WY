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

from .models import IntentError, uid
from .prompts import build_messages, parse_samples
from .providers import ProviderError


class JobRunner:
    """Browser-independent workers; every committed batch is durable and idempotent."""

    def __init__(self, store, retry_delay=2.0):
        self.store, self.retry_delay = store, retry_delay
        self._lock = threading.Lock()
        self._threads = {}
        self._wake = {}

    def start(self, job_id, provider):
        with self._lock:
            if job_id in self._threads and self._threads[job_id].is_alive():
                raise IntentError("任务已经在运行。")
            self.store.claim_job(job_id, provider.descriptor)
            event = self._wake[job_id] = threading.Event()
            worker = threading.Thread(
                target=self._run, args=(job_id, provider, event), daemon=True, name=f"intent-{job_id[:8]}"
            )
            self._threads[job_id] = worker
            worker.start()

    def pause(self, job_id):
        if not self.store.transition(job_id, ("running",), "pausing"):
            raise IntentError("只有运行中的任务可以暂停。")
        self._signal(job_id)

    def cancel(self, job_id):
        if not self.store.transition(job_id, ("running", "pausing"), "cancelling"):
            if not self.store.transition(job_id, ("queued", "paused", "interrupted", "failed"), "cancelled"):
                raise IntentError("任务已经结束。")
        self._signal(job_id)

    def _signal(self, job_id):
        with self._lock:
            if job_id in self._wake:
                self._wake[job_id].set()

    def wait(self, job_id, timeout=30):
        with self._lock:
            worker = self._threads.get(job_id)
        if worker:
            worker.join(timeout)
        return self.store.job(job_id)

    def _run(self, job_id, provider, event):
        try:
            with provider.session():
                self._loop(job_id, provider, event)
        except IntentError as error:
            self.store.transition(job_id, ("running", "pausing", "cancelling"), "failed", str(error))
        except Exception:
            # Never persist arbitrary HTTP/library exception strings which may contain credentials.
            self.store.transition(
                job_id,
                ("running", "pausing", "cancelling"),
                "failed",
                "生成任务发生内部异常，已保留完成的批次。请检查服务环境后继续。",
            )

    def _loop(self, job_id, provider, event):
        consecutive_errors = 0
        while True:
            job = self.store.job(job_id)
            if job["status"] in {"pausing", "cancelling"}:
                target = "paused" if job["status"] == "pausing" else "cancelled"
                self.store.transition(job_id, (job["status"],), target)
                return
            if job["status"] != "running":
                return
            remaining = {
                k: v - job["by_label"].get(k, 0) for k, v in job["targets"].items() if v > job["by_label"].get(k, 0)
            }
            if not remaining:
                self.store.transition(job_id, ("running",), "completed")
                return
            if job["empty_batches"] >= 5 or job["attempts"] >= job["max_attempts"]:
                self.store.transition(
                    job_id, ("running",), "failed", "已达到无有效产出或尝试次数上限，请改进样例或配置后再继续。"
                )
                return
            label = list(remaining)[job["attempts"] % len(remaining)]
            intents = self.store.version(job["version_id"])["config"]
            target = next(x for x in intents if x["label"] == label)
            count = min(provider.config["batch_size"], remaining[label])
            usage, previous = self.store.seed_usage(job_id, label)
            messages, seed_ids = build_messages(
                intents,
                target,
                self.store.seeds(job["version_id"], label),
                count,
                job["attempts"],
                provider.config["seed_count"],
                usage,
                previous,
                job_id,
            )
            error, fatal, produced, invalid, texts = "", False, 0, 0, []
            try:
                texts, produced, invalid = parse_samples(provider.generate(messages), count, target)
                consecutive_errors = 0
            except ProviderError as failure:
                error = str(failure)
                consecutive_errors += 1
                fatal = not failure.retryable or consecutive_errors >= 3
            except IntentError as failure:
                error = str(failure)
                invalid, consecutive_errors = 1, consecutive_errors + 1
                fatal = consecutive_errors >= 3
            self.store.commit_batch(job_id, uid(), label, texts, seed_ids, produced, invalid, error)
            # A requested pause/cancel takes precedence over retry failure after an in-flight request.
            if self.store.job(job_id)["status"] in {"pausing", "cancelling"}:
                continue
            if fatal:
                self.store.transition(job_id, ("running",), "failed", error)
                return
            if error:
                event.wait(self.retry_delay * consecutive_errors)
                event.clear()
