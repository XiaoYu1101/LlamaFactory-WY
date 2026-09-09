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
from contextlib import nullcontext

import gradio as gr
import pytest

from llamafactory.intent.providers import DEFAULT_CONFIG
from llamafactory.intent.service import IntentService
from llamafactory.intent.ui import create_intent_page, selection


def test_ui_configuration_generation_review_and_export(tmp_path, monkeypatch):
    class Provider:
        config = dict(DEFAULT_CONFIG)
        descriptor = {"kind": "test", "model": "ui-test"}
        number = 0

        def session(self):
            return nullcontext()

        def generate(self, messages):
            count = json.loads(messages[1]["content"])["生成数量"]
            items = [f"界面测试新样本 {self.number + i}" for i in range(count)]
            self.number += count
            return json.dumps({"samples": items})

    service = IntentService(tmp_path / "data")
    monkeypatch.setattr(service, "provider", lambda chatter=None: Provider())
    try:
        with gr.Blocks() as ui:
            elements = create_intent_page(service=service)
        assert elements["project"].value is None
        assert elements["config"].interactive is True
        assert elements["table"].static_columns == [1, 2, 3, 4, 5, 6]
        callbacks = {block.api_name: block.fn for block in ui.fns.values() if block.fn}
        assert len(callbacks["load_project"](None)) == 11
        assert callbacks["refresh_samples"]("", "", "", "", 1, False)[0] == []
        saved = callbacks["save_project"]("界面项目", [["refund", "退款", "查询退款进度", "何时到账？", 12]], None)
        loaded = callbacks["load_project"](saved["value"])
        version_id = loaded[2]
        created, config_info = callbacks["start_generation"](version_id)
        assert service.jobs.wait(created["value"])["accepted"] == 12
        view, stored, page, info, counts = callbacks["refresh_samples"](version_id, "", "", "", 1, False)
        assert len(view) == 13
        assert len(view[0]) == 7
        view[1][0] = True
        editor = callbacks["load_editor"](view, stored)
        callbacks["edit_sample"](version_id, editor[0], editor[1], editor[2], "人工修正后的问法")
        # An old selection must not approve an edited record.
        with pytest.raises(gr.Error):
            callbacks["review_approved"](version_id, view, stored)
        view, stored, *_ = callbacks["refresh_samples"](version_id, "", "", "", 1, False)
        for row in view:
            row[0] = True
        callbacks["review_approved"](version_id, view, stored)
        export = callbacks["freeze_version"](version_id)
        description, file = callbacks["show_export"](export["value"])
        assert "13" in description and file.endswith("dataset.zip")
        assert callbacks["poll"](created["value"])[0] == 100
    finally:
        service.lock.close()


def test_selection_cannot_substitute_record_ids():
    with pytest.raises(ValueError):
        selection([[True, "different-record"]], [{"id": "original", "revision": 1}])
