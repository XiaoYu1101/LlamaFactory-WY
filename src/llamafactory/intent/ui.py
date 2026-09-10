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

from functools import wraps

import gradio as gr

from .export import freeze_dataset, get_export
from .models import IntentError, integer
from .providers import read_config
from .service import get_service


STATE_NAMES = {
    "pending": "待审核",
    "approved": "已通过",
    "rejected": "已驳回",
    "queued": "排队中",
    "running": "生成中",
    "pausing": "正在暂停",
    "paused": "已暂停",
    "completed": "已完成",
    "failed": "失败",
    "cancelled": "已停止",
    "cancelling": "正在停止",
    "interrupted": "已中断",
}
SOURCE_NAMES = {"user": "用户样例", "manual": "人工添加", "generated": "模型生成"}
CONFIG_HEADERS = ["类别标识", "类别名称", "类别说明", "样例（每行一条）", "新增数量"]
SAMPLE_HEADERS = ["选择", "ID", "类别", "文本", "审核状态", "来源", "标签冲突"]


def guarded(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except IntentError as error:
            raise gr.Error(str(error)) from None

    return wrapped


def config_rows(intents):
    return [[x["label"], x["name"], x["description"], "\n".join(x["examples"]), x["target"]] for x in intents]


def rows_config(rows):
    return [
        dict(zip(("label", "name", "description", "examples", "target"), row))
        for row in (rows or [])
        if any(str(x or "").strip() for x in row)
    ]


def selection(table, snapshot):
    selected = []
    for index, row in enumerate(table or []):
        if row and row[0] is True:
            if index >= len(snapshot) or row[1] != snapshot[index]["id"]:
                raise IntentError("列表已经变化，请刷新后重新选择。")
            selected.append({"id": snapshot[index]["id"], "revision": snapshot[index]["revision"]})
    if not selected:
        raise IntentError("请先勾选需要处理的样本。")
    return selected


def create_intent_page(engine=None, service=None):
    service = service or get_service()
    store = service.store
    chatter = engine.chatter if engine else None
    gr.Markdown("## 意图分类数据工作台\n配置类别与样例，分批生成数据，经人工审核后用于微调。")

    def project_choices():
        return [(f"{x['name']} · {x['id'][:6]}", x["id"]) for x in store.projects()]

    with gr.Row():
        project = gr.Dropdown(label="已保存的项目", choices=project_choices(), value=None, scale=4)
        new_project = gr.Button("新建项目", scale=1)
        refresh_projects = gr.Button("刷新项目", scale=1)
    project_name = gr.Textbox(label="项目名称", placeholder="例如：售后客服意图分类")
    version = gr.State("")
    snapshot = gr.State([])

    with gr.Accordion("1 · 配置意图", open=True):
        config = gr.Dataframe(
            headers=CONFIG_HEADERS,
            datatype=["str", "str", "str", "str", "number"],
            type="array",
            row_count=(1, "dynamic"),
            col_count=(5, "fixed"),
            value=[["", "", "", "", 100]],
            interactive=True,
            wrap=True,
            label="每类至少一条样例；使用表格的添加行按钮增加类别",
        )
        with gr.Row():
            total = gr.Number(label="可选：将总量均分到各类别", value=1000, precision=0)
            distribute = gr.Button("均分到表格")
            save = gr.Button("保存配置", variant="primary")
        config_status = gr.Markdown("保存后才能生成。修改类别定义会创建新版本，旧数据版本继续保留。")

    with gr.Accordion("2 · 分批生成", open=True):
        provider_info = gr.Markdown()
        with gr.Row():
            generate = gr.Button("按已保存配额生成", variant="primary")
            job = gr.Dropdown(label="生成任务", choices=[], scale=4)
            resume = gr.Button("继续")
            pause = gr.Button("暂停")
            cancel = gr.Button("停止")
        with gr.Row():
            extra_label = gr.Dropdown(label="按类别补生成", choices=[])
            extra_amount = gr.Number(label="补生成数量", value=10, precision=0)
            supplement = gr.Button("补生成")
        progress = gr.Slider(0, 100, value=0, label="生成进度 %", interactive=False)
        job_status = gr.Markdown("尚未启动生成。")
        timer = gr.Timer(2)

    with gr.Accordion("3 · 人工审核", open=True):
        counts = gr.Markdown()
        with gr.Row():
            label_filter = gr.Dropdown(label="类别", choices=[("全部", "")], value="")
            status_filter = gr.Dropdown(
                label="状态",
                choices=[("全部", ""), ("待审核", "pending"), ("已通过", "approved"), ("已驳回", "rejected")],
                value="",
            )
            query = gr.Textbox(label="搜索文本")
            deleted = gr.Checkbox(label="查看已删除数据", value=False)
        with gr.Row():
            page = gr.Number(label="页码", value=1, precision=0)
            previous = gr.Button("上一页")
            next_page = gr.Button("下一页")
            refresh = gr.Button("刷新列表")
        table = gr.Dataframe(
            headers=SAMPLE_HEADERS,
            datatype=["bool", "str", "str", "str", "str", "str", "str"],
            type="array",
            col_count=(7, "fixed"),
            row_count=(0, "fixed"),
            value=[],
            interactive=True,
            static_columns=[1, 2, 3, 4, 5, 6],
            wrap=True,
            column_widths=[60, 140, 130, 420, 100, 100, 100],
            label="勾选后批量处理；编辑文本请使用下方编辑区",
        )
        page_info = gr.Markdown()
        with gr.Row():
            check_page = gr.Button("勾选本页")
            approve = gr.Button("通过选中", variant="primary")
            reject = gr.Button("驳回选中")
            delete = gr.Button("删除选中")
            restore = gr.Button("恢复选中")
        review_status = gr.Markdown()
        with gr.Row():
            edit_id = gr.Textbox(label="编辑中的样本 ID", interactive=False)
            edit_revision = gr.State(0)
            edit_label = gr.Dropdown(label="所属类别", choices=[])
            load_edit = gr.Button("将勾选的一条载入编辑区")
        edit_text = gr.Textbox(label="样本文本", lines=3)
        save_edit = gr.Button("保存修改并重新审核")
        with gr.Accordion("人工补充样本", open=False):
            manual_label = gr.Dropdown(label="类别", choices=[])
            manual_text = gr.Textbox(label="新样本文本", lines=2)
            add_manual = gr.Button("添加为待审核")

    with gr.Accordion("4 · 数据版本与微调", open=True):
        gr.Markdown("所有待审核项处理完毕后，保存不可变的数据版本。导出只包含已通过数据。")
        freeze = gr.Button("完成审核并保存数据版本", variant="primary")
        export_choice = gr.Dropdown(label="已保存的数据版本", choices=[])
        export_info = gr.Markdown()
        download = gr.File(
            label="下载数据集（含 train.json、dataset_info.json、类别定义及审核快照）", interactive=False
        )
        if engine:
            train = gr.Button("使用内置参数开始微调", variant="primary")
            gr.Markdown(
                "先在顶部选择训练模型。训练日志、进度和结果显示在原 Train 页面；评测使用原 Evaluate & Predict 页面。"
            )
        else:
            train = None
            gr.Markdown(
                "当前为无需本地模型的数据工作台。下载数据后，在完整 LlamaFactory WebUI 中选择数据目录进行微调；"
                "使用同一工作区启动完整 WebUI 时，也可直接在此页面启动训练。"
            )

    def config_info():
        cfg = read_config(service.config_path)
        return (
            f"生成方式：**{cfg['provider']}** · 模型：**{cfg['model']}** · "
            f"每批最多 **{cfg['seed_count']}** 条参考样例 / **{cfg['batch_size']}** 条新数据。\n\n"
            "连接设置来自生成配置文件，密钥读取本机环境变量或 .env。每批为独立请求。"
        )

    def job_choices(version_id):
        return (
            [
                (f"{x['id'][:8]} · {STATE_NAMES.get(x['status'], x['status'])} · {x['accepted']} 条", x["id"])
                for x in store.jobs(version_id)
            ]
            if version_id
            else []
        )

    def export_choices(version_id):
        return [(f"{x['created']} · {x['id'][:8]}", x["id"]) for x in store.exports(version_id)] if version_id else []

    @guarded
    def load_project(project_id):
        if not project_id:
            return (
                "",
                [["", "", "", "", 100]],
                "",
                gr.update(choices=[("全部", "")], value=""),
                *[gr.update(choices=[], value=None) for _ in range(5)],
                "新项目，请填写并保存。",
                config_info(),
            )
        item = store.project(project_id)
        intents, version_id = item["version"]["config"], item["current_version"]
        choices = [(f"{x['name']} ({x['label']})", x["label"]) for x in intents]
        jobs, exports = job_choices(version_id), export_choices(version_id)
        return (
            item["name"],
            config_rows(intents),
            version_id,
            gr.update(choices=[("全部", "")] + choices, value=""),
            *[gr.update(choices=choices, value=choices[0][1]) for _ in range(3)],
            gr.update(choices=jobs, value=jobs[0][1] if jobs else None),
            gr.update(choices=exports, value=exports[0][1] if exports else None),
            f"配置版本：{version_id[:8]}。生成使用已保存配置；定义变化后旧样本不会自动移入新版本。",
            config_info(),
        )

    load_outputs = [
        project_name,
        config,
        version,
        label_filter,
        edit_label,
        manual_label,
        extra_label,
        job,
        export_choice,
        config_status,
        provider_info,
    ]

    @guarded
    def save_project(name, rows, project_id):
        item = store.save_project(name, rows_config(rows), project_id)
        return gr.update(choices=project_choices(), value=item["id"])

    @guarded
    def distribute_total(rows, amount):
        intents = rows_config(rows)
        amount = integer(amount, "总量", minimum=max(1, len(intents)))
        if not intents:
            raise IntentError("请先填写类别。")
        quotient, remainder = divmod(amount, len(intents))
        for index, row in enumerate(intents):
            row["target"] = quotient + (index < remainder)
            if isinstance(row["examples"], str):
                row["examples"] = row["examples"].splitlines()
        return config_rows(intents)

    @guarded
    def refresh_samples(version_id, label, status, text_query, page_number, show_deleted):
        if not version_id:
            return [], [], 1, "请先选择或保存项目。", ""
        rows, total_rows, current = store.samples(
            version_id, label, status, text_query, page_number, deleted=show_deleted
        )
        view = [
            [
                False,
                x["id"],
                x["label"],
                x["text"],
                STATE_NAMES[x["status"]],
                SOURCE_NAMES[x["source"]],
                "待解决" if x["conflict"] else "",
            ]
            for x in rows
        ]
        stats = store.counts(version_id)
        groups = stats["groups"]
        numbers = {
            state: sum(x["n"] for x in groups if x["status"] == state and not x["deleted"])
            for state in ("pending", "approved", "rejected")
        }
        summary = " · ".join(f"{STATE_NAMES[k]} **{v}**" for k, v in numbers.items())
        summary += f" · 标签冲突 **{stats['conflicts']}** 组"
        return (
            view,
            rows,
            current,
            f"第 {current} / {max(1, (total_rows + 24) // 25)} 页，共 {total_rows} 条。",
            summary,
        )

    refresh_inputs = [version, label_filter, status_filter, query, page, deleted]
    refresh_outputs = [table, snapshot, page, page_info, counts]

    @guarded
    def start_generation(version_id, label=None, amount=None):
        if not version_id:
            raise IntentError("请先保存配置。")
        job_id = service.start(version_id, chatter, label, amount)
        return gr.update(choices=job_choices(version_id), value=job_id), config_info()

    @guarded
    def poll(job_id):
        if not job_id:
            return 0, "尚未启动生成。"
        item = store.job(job_id)
        target = sum(item["targets"].values())
        info = (
            f"**{STATE_NAMES.get(item['status'], item['status'])}** · 有效新增 {item['accepted']} / {target} · "
            f"重复 {item['duplicates']} · 异常条目 {item['invalid']} · 已尝试 {item['attempts']} 批"
        )
        if item["error"]:
            info += "\n\n" + item["error"]
        return round(100 * item["accepted"] / target, 1), info

    @guarded
    def control_job(job_id, action):
        if not job_id:
            raise IntentError("请先选择生成任务。")
        if action == "resume":
            service.jobs.start(job_id, service.provider(chatter))
        elif action == "pause":
            service.jobs.pause(job_id)
        else:
            service.jobs.cancel(job_id)
        return poll(job_id)

    @guarded
    def apply_review(version_id, rows, stored, action):
        selected = selection(rows, stored)
        store.review(version_id, selected, action)
        return f"已处理选中的 {len(selected)} 条样本。"

    @guarded
    def load_editor(rows, stored):
        selected = selection(rows, stored)
        if len(selected) != 1:
            raise IntentError("请只勾选一条样本进行编辑。")
        item = next(x for x in stored if x["id"] == selected[0]["id"])
        return item["id"], item["revision"], item["label"], item["text"]

    @guarded
    def edit_sample(version_id, sample_id, revision, label, text):
        if not sample_id:
            raise IntentError("请先载入要编辑的样本。")
        store.review(version_id, [{"id": sample_id, "revision": revision}], "edit", text=text, label=label)
        return "修改已保存，该样本需要重新审核。", "", 0, ""

    @guarded
    def add_sample(version_id, label, text):
        store.add_sample(version_id, label, text)
        return "已添加为待审核样本。", ""

    @guarded
    def freeze_version(version_id):
        manifest = freeze_dataset(store, version_id)
        return gr.update(choices=export_choices(version_id), value=manifest["id"])

    @guarded
    def show_export(export_id):
        if not export_id:
            return "尚无已保存的数据版本。", None
        item = get_export(store, export_id)
        return (
            f"固定版本 **{export_id[:8]}** · **{item['manifest']['count']}** 条已通过样本。"
            "\n\n未自动划分评测集，请另外准备独立测试数据。",
            str(store.root / "datasets" / export_id / "dataset.zip"),
        )

    project.input(load_project, [project], load_outputs).then(refresh_samples, refresh_inputs, refresh_outputs)
    new_project.click(lambda: gr.update(value=None), outputs=project).then(load_project, [project], load_outputs).then(
        refresh_samples, refresh_inputs, refresh_outputs
    )
    refresh_projects.click(lambda: gr.update(choices=project_choices()), outputs=project).then(
        load_project, [project], load_outputs
    ).then(refresh_samples, refresh_inputs, refresh_outputs)
    save.click(save_project, [project_name, config, project], project).then(
        load_project, [project], load_outputs
    ).then(refresh_samples, refresh_inputs, refresh_outputs)
    distribute.click(distribute_total, [config, total], config)
    generate.click(start_generation, [version], [job, provider_info])
    supplement.click(start_generation, [version, extra_label, extra_amount], [job, provider_info])
    for button, action in ((resume, "resume"), (pause, "pause"), (cancel, "cancel")):
        button.click(lambda job_id, action=action: control_job(job_id, action), [job], [progress, job_status])
    timer.tick(poll, [job], [progress, job_status], show_progress="hidden", queue=False)
    job.change(poll, [job], [progress, job_status])
    refresh.click(refresh_samples, refresh_inputs, refresh_outputs)
    for component in (label_filter, status_filter, deleted):
        component.input(lambda: 1, outputs=page).then(refresh_samples, refresh_inputs, refresh_outputs)
    previous.click(lambda x: max(1, int(x or 1) - 1), [page], [page]).then(
        refresh_samples, refresh_inputs, refresh_outputs
    )
    next_page.click(lambda x: int(x or 1) + 1, [page], [page]).then(refresh_samples, refresh_inputs, refresh_outputs)
    check_page.click(lambda rows: [[True, *row[1:]] for row in rows], [table], [table])
    for button, action in ((approve, "approved"), (reject, "rejected"), (delete, "delete"), (restore, "restore")):
        button.click(
            lambda v, rows, stored, action=action: apply_review(v, rows, stored, action),
            [version, table, snapshot],
            [review_status],
            api_name=f"review_{action}",
        ).then(refresh_samples, refresh_inputs, refresh_outputs)
    load_edit.click(load_editor, [table, snapshot], [edit_id, edit_revision, edit_label, edit_text])
    save_edit.click(
        edit_sample,
        [version, edit_id, edit_revision, edit_label, edit_text],
        [review_status, edit_id, edit_revision, edit_text],
    ).then(refresh_samples, refresh_inputs, refresh_outputs)
    add_manual.click(add_sample, [version, manual_label, manual_text], [review_status, manual_text]).then(
        refresh_samples, refresh_inputs, refresh_outputs
    )
    freeze.click(freeze_version, [version], [export_choice])
    export_choice.change(show_export, [export_choice], [export_info, download])

    if train is not None:
        from .training import attach_training, training_outputs

        train_inputs = set(engine.runner.train_input_elems) | {export_choice}
        train_outputs = training_outputs(engine)

        def start_training(data):
            try:
                if not data[export_choice]:
                    raise IntentError("请先完成审核并选择数据版本。")
                yield from attach_training(engine, store, data[export_choice], data)
            except IntentError as error:
                raise gr.Error(str(error)) from None

        train.click(start_training, train_inputs, train_outputs)
    return {
        "project": project,
        "config": config,
        "table": table,
        "version": version,
        "job": job,
        "provider_info": provider_info,
        "config_info": config_info,
    }


def create_standalone(root=None, config_path=None):
    service = get_service(root, config_path)
    with gr.Blocks(title="LlamaFactory-WY · 意图分类", theme=gr.themes.Soft()) as demo:
        elements = create_intent_page(service=service)
        demo.load(guarded(elements["config_info"]), outputs=elements["provider_info"])
    return demo


def create_ant_page(engine):
    """Embed the Ant Design workspace; keep training on the original Gradio event queue."""
    from .training import training_outputs

    service = get_service()
    gr.HTML(
        '<iframe src="intent/" title="意图分类工作台" style="width:100%;height:1050px;border:0;border-radius:12px"></iframe>'
    )
    gr.Markdown("### 使用已审核数据开始微调\n先在顶部选择模型；训练状态、结果和评测使用原有页面。")
    with gr.Row():
        export_choice = gr.Dropdown(label="已保存的数据版本", choices=[], value=None, scale=4)
        refresh = gr.Button("刷新数据版本")
        prepare = gr.Button("将数据填入训练页")
        train = gr.Button("使用内置参数开始微调", variant="primary")

    def choices():
        with service.store.connect() as db:
            rows = db.execute(
                "SELECT e.id,e.created,p.name FROM exports e JOIN versions v ON e.version_id=v.id "
                "JOIN projects p ON v.project_id=p.id ORDER BY e.created DESC,e.rowid DESC"
            ).fetchall()
        return gr.update(choices=[(f"{x['name']} · {x['created']} · {x['id'][:8]}", x["id"]) for x in rows])

    def start_training(data):
        from .training import attach_training

        try:
            if not data[export_choice]:
                raise IntentError("请先刷新并选择已保存的数据版本。")
            yield from attach_training(engine, service.store, data[export_choice], data)
        except IntentError as error:
            raise gr.Error(str(error)) from None

    def prepare_training(export_id):
        from ..extras.constants import TRAINING_STAGES
        from .export import get_export
        from .training import training_values

        if engine.runner.running:
            raise gr.Error("已有训练或评测任务运行，不能替换参数。")
        if not export_id:
            raise gr.Error("请先刷新并选择已保存的数据版本。")
        export = get_export(service.store, export_id)
        if export["manifest"].get("training_ready") is False:
            raise gr.Error(export["manifest"]["training_note"])
        values = training_values(export, TRAINING_STAGES, "fp32")
        updates = {engine.manager.get_elem_by_id(name): gr.update(value=value) for name, value in values.items()}
        updates[engine.manager.get_elem_by_id("top.checkpoint_path")] = gr.update(value=[], multiselect=True)
        updates[engine.intent_tabs] = gr.update(selected="train")
        return updates

    gr.Markdown(
        "### 已保存的 LoRA 配置\n在工作台模型训练页保存配置后，点击刷新。顶部先选模型名称（自定义路径可选 Custom），再填入配置；填入后到 Train 核对并开始。"
    )
    with gr.Row():
        plan_choice = gr.Dropdown(label="已保存的 LoRA 配置", choices=[], value=None, scale=4)
        refresh_plans = gr.Button("刷新 LoRA 配置")
        prepare_plan = gr.Button("将 LoRA 配置填入训练页")

    def plan_choices():
        from .training_config import list_plans

        return gr.update(
            choices=[
                (f"{p['config']['model_name_or_path']} · {p['created']} · {p['id'][:8]}", p["id"])
                for p in list_plans(service.store)
            ]
        )

    def prepare_saved_plan(plan_id, model_name):
        import torch

        from ..extras.constants import TRAINING_STAGES
        from .training import configured_training_values
        from .training_config import LoraConfig, get_plan, validate_plan

        if engine.runner.running:
            raise gr.Error("已有训练或评测任务运行，不能替换参数。")
        if not model_name:
            raise gr.Error("请先在顶部选择模型名称；自定义模型可选择 Custom。")
        try:
            plan = get_plan(service.store, plan_id)
            config = LoraConfig(**plan["config"])
            export = validate_plan(service.store, plan["export_id"], config)
        except IntentError as error:
            raise gr.Error(str(error)) from None
        precision = (
            "bf16"
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else "fp16"
            if torch.cuda.is_available()
            else "fp32"
        )
        values = configured_training_values(export, TRAINING_STAGES, precision, config, plan_id)
        # Reset prior advanced training options so a previous session cannot silently enable another method.
        base = engine.manager.get_base_elems()
        updates = {elem: gr.update(value=elem.value) for elem in engine.runner.train_input_elems if elem not in base}
        updates.update({engine.manager.get_elem_by_id(name): gr.update(value=value) for name, value in values.items()})
        updates[engine.manager.get_elem_by_id("top.checkpoint_path")] = gr.update(value=[], multiselect=True)
        updates[engine.intent_tabs] = gr.update(selected="train")
        return updates

    refresh_plans.click(plan_choices, outputs=plan_choice)
    prepare_plan.click(
        prepare_saved_plan,
        inputs=[plan_choice, engine.manager.get_elem_by_id("top.model_name")],
        outputs=training_outputs(engine) + [engine.intent_tabs],
    )

    prepare.click(prepare_training, inputs=export_choice, outputs=training_outputs(engine) + [engine.intent_tabs])
    refresh.click(choices, outputs=export_choice)
    train.click(
        start_training,
        set(engine.runner.train_input_elems) | {export_choice},
        training_outputs(engine),
    )

    gr.Markdown(
        "### 填入独立评测数据\n选择另行准备并审核导出的测试数据版本。这里只填入数据，不自动开始评测；顶部保留你选择的基座模型和 LoRA 适配器。不要将训练数据当作独立测试集。"
    )
    with gr.Row():
        eval_export = gr.Dropdown(label="独立评测数据版本", choices=[], value=None, scale=4)
        refresh_eval = gr.Button("刷新评测数据版本")
        prepare_eval = gr.Button("将数据填入评测页")

    def prepare_evaluation(export_id):
        from .export import get_export
        from .models import uid

        if engine.runner.running:
            raise gr.Error("已有训练或评测任务运行，不能替换参数。")
        if not export_id:
            raise gr.Error("请选择独立评测数据版本。")
        try:
            export = get_export(service.store, export_id)
        except IntentError as error:
            raise gr.Error(str(error)) from None
        if not export["manifest"].get("training_ready"):
            raise gr.Error("评测数据缺少问题与参考答案的字段映射，请配置映射后重新导出。")
        values = {
            "eval.dataset_dir": export["path"],
            "eval.dataset": export["manifest"]["training_datasets"],
            "eval.max_samples": str(export["manifest"]["count"]),
            "eval.output_dir": f"wy_eval_{export_id[:8]}_{uid()[:8]}",
        }
        updates = {engine.manager.get_elem_by_id(name): gr.update(value=value) for name, value in values.items()}
        updates[engine.intent_tabs] = gr.update(selected="eval")
        return updates

    refresh_eval.click(choices, outputs=eval_export)
    prepare_eval.click(
        prepare_evaluation,
        inputs=eval_export,
        outputs=[
            engine.manager.get_elem_by_id("eval." + key)
            for key in ("dataset_dir", "dataset", "max_samples", "output_dir")
        ]
        + [engine.intent_tabs],
    )
