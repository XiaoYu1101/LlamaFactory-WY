# Copyright 2025 the LlamaFactory team.
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

import os
import platform

from ..extras.misc import fix_proxy, is_env_enabled
from ..extras.packages import is_gradio_available
from ..intent.ui import create_ant_page
from .common import save_config
from .components import (
    create_chat_box,
    create_eval_tab,
    create_export_tab,
    create_footer,
    create_infer_tab,
    create_top,
    create_train_tab,
)
from .css import CSS
from .engine import Engine


if is_gradio_available():
    import gradio as gr


def create_ui(demo_mode: bool = False) -> "gr.Blocks":
    engine = Engine(demo_mode=demo_mode, pure_chat=False)
    hostname = os.getenv("HOSTNAME", os.getenv("COMPUTERNAME", platform.node())).split(".")[0]

    with gr.Blocks(title=f"LLaMA Factory ({hostname})", css=CSS) as demo:
        title = gr.HTML()
        subtitle = gr.HTML()
        if demo_mode:
            gr.DuplicateButton(value="Duplicate Space for private use", elem_classes="duplicate-button")

        engine.manager.add_elems("head", {"title": title, "subtitle": subtitle})
        engine.manager.add_elems("top", create_top())
        lang: gr.Dropdown = engine.manager.get_elem_by_id("top.lang")

        with gr.Tabs() as workflow_tabs:
            engine.intent_tabs = workflow_tabs
            with gr.Tab("Train", id="train", elem_id="wy-native-train"):
                engine.manager.add_elems("train", create_train_tab(engine))

            with gr.Tab("Evaluate & Predict", id="eval", elem_id="wy-native-eval"):
                engine.manager.add_elems("eval", create_eval_tab(engine))

            with gr.Tab("Chat"):
                engine.manager.add_elems("infer", create_infer_tab(engine))

            if not demo_mode:
                with gr.Tab("意图分类", id="intent"):
                    create_ant_page(engine)

                with gr.Tab("Export"):
                    engine.manager.add_elems("export", create_export_tab(engine))

        engine.manager.add_elems("footer", create_footer())
        demo.load(engine.resume, outputs=engine.manager.get_elem_list(), concurrency_limit=None)
        demo.load(
            fn=None,
            js="""() => {
            const navigate = (tab) => {
                if (!["train", "eval"].includes(tab)) return;
                const button = document.getElementById(`wy-native-${tab}-button`);
                if (button) { button.click(); window.scrollTo({top: 0, behavior: "smooth"}); }
            };
            if (window.__wyNavigate) window.removeEventListener("message", window.__wyNavigate);
            window.__wyNavigate = (event) => {
                const frame = document.querySelector('iframe[title="意图分类工作台"]');
                if (event.origin === window.location.origin && event.source === frame?.contentWindow && event.data?.type === "wy-native-tab") navigate(event.data.tab);
            };
            window.addEventListener("message", window.__wyNavigate);
            navigate(new URLSearchParams(window.location.search).get("wy_tab"));
        }""",
        )
        lang.change(engine.change_lang, [lang], engine.manager.get_elem_list(), queue=False)
        lang.input(save_config, inputs=[lang], queue=False)

    demo.intent_engine = engine
    return demo


def create_web_demo() -> "gr.Blocks":
    engine = Engine(pure_chat=True)
    hostname = os.getenv("HOSTNAME", os.getenv("COMPUTERNAME", platform.node())).split(".")[0]

    with gr.Blocks(title=f"LLaMA Factory Web Demo ({hostname})", css=CSS) as demo:
        lang = gr.Dropdown(choices=["en", "ru", "zh", "ko", "ja"], scale=1)
        engine.manager.add_elems("top", dict(lang=lang))

        _, _, chat_elems = create_chat_box(engine, visible=True)
        engine.manager.add_elems("infer", chat_elems)

        demo.load(engine.resume, outputs=engine.manager.get_elem_list(), concurrency_limit=None)
        lang.change(engine.change_lang, [lang], engine.manager.get_elem_list(), queue=False)
        lang.input(save_config, inputs=[lang], queue=False)

    return demo


def run_web_ui() -> None:
    gradio_ipv6 = is_env_enabled("GRADIO_IPV6")
    gradio_share = is_env_enabled("GRADIO_SHARE")
    server_name = os.getenv("GRADIO_SERVER_NAME", "[::]" if gradio_ipv6 else "0.0.0.0")
    print("[1/3] 正在初始化原生训练、评测界面；此时尚未监听端口。", flush=True)
    fix_proxy(ipv6_enabled=gradio_ipv6)
    from ..intent.api import install_routes
    from ..intent.service import get_service

    demo = create_ui().queue()
    print("[2/3] 界面初始化完成，正在启动 HTTP 服务。", flush=True)
    demo.launch(share=gradio_share, server_name=server_name, inbrowser=False, prevent_thread_lock=True)
    install_routes(demo.app, get_service(), demo.intent_engine)
    display_host = "<服务器IP>" if server_name in {"0.0.0.0", "::", "[::]"} else server_name
    print(f"[3/3] 意图工作台路由已就绪：http://{display_host}:{demo.server_port}/intent/", flush=True)
    demo.block_thread()


def run_web_demo() -> None:
    gradio_ipv6 = is_env_enabled("GRADIO_IPV6")
    gradio_share = is_env_enabled("GRADIO_SHARE")
    server_name = os.getenv("GRADIO_SERVER_NAME", "[::]" if gradio_ipv6 else "0.0.0.0")
    print("Visit http://ip:port for Web UI, e.g., http://127.0.0.1:7860")
    fix_proxy(ipv6_enabled=gradio_ipv6)
    create_web_demo().queue().launch(share=gradio_share, server_name=server_name, inbrowser=True)
