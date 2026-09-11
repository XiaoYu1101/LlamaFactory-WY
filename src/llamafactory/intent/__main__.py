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

import argparse
import os


def main():
    parser = argparse.ArgumentParser(description="无需本地模型的意图分类数据工作台")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--data-dir", default=os.getenv("LF_INTENT_HOME", "workspace/intent"))
    parser.add_argument("--config", default=os.getenv("LF_INTENT_CONFIG", "config/intent_generation.json"))
    parser.add_argument("--full", action="store_true", help="启动含原生训练、评测、Chat 和数据工作台的完整 WebUI")
    args = parser.parse_args()
    if args.full:
        os.environ.update(
            LF_INTENT_HOME=args.data_dir,
            LF_INTENT_CONFIG=args.config,
            GRADIO_SERVER_NAME=args.host,
            GRADIO_SERVER_PORT=str(args.port),
        )
        try:
            from ..webui.interface import run_web_ui
        except ImportError as error:
            parser.exit(
                1, f"完整模式缺少依赖：{error}\n请先执行 python -m pip install -e . 并配置模型所需的训练环境。\n"
            )
        run_web_ui()
        return
    import uvicorn

    from .api import create_app

    uvicorn.run(create_app(args.data_dir, args.config), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
