"""Check WebUI dependencies and initialization without binding a port or loading a model."""

import faulthandler
import importlib.metadata
import platform
import sys
import time


def main():
    print("Python:", sys.version, flush=True)
    print("Executable:", sys.executable, flush=True)
    print("Platform:", platform.platform(), platform.machine(), flush=True)
    for name in ("gradio", "fastapi", "starlette", "pydantic", "torch", "transformers"):
        try:
            value = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            value = "NOT INSTALLED"
        print(f"{name}: {value}", flush=True)
    print("No server or model will be started. Stack traces print every 30s if initialization is slow.", flush=True)
    faulthandler.dump_traceback_later(30, repeat=True)
    started = time.monotonic()
    try:
        print("[1/3] Importing Pydantic v1 compatibility...", flush=True)
        from pydantic import v1

        print(f"Pydantic compatibility ready ({time.monotonic() - started:.2f}s): {v1.VERSION}", flush=True)
        print("[2/3] Importing Gradio...", flush=True)
        import gradio as gr

        print(f"Gradio imported ({time.monotonic() - started:.2f}s)", flush=True)
        print("[3/3] Initializing a minimal Gradio Blocks and queue (no launch)...", flush=True)
        with gr.Blocks(analytics_enabled=False) as demo:
            gr.Markdown("Dependency initialization check")
        demo.queue()
        print(f"PASS: Web dependency initialization completed in {time.monotonic() - started:.2f}s.", flush=True)
        print("This does not test the full project startup, GPU, model service or LAN connectivity.", flush=True)
    finally:
        faulthandler.cancel_dump_traceback_later()


if __name__ == "__main__":
    main()
