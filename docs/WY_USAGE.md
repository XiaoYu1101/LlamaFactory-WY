# JSON 样例扩写工作台

## 工作方式

一个“生成场景”代表一项数据扩写任务。例如 `dam_intent` 是大坝监测数据场景的标识，**不是模型的输出标签**。把完整的 `instruction` / `input` / `output` 样例交给模型，生成新的问句及对应答案，再审核、导出、微调。

支持 JSON 数组、单个 JSON 对象和 JSONL（每行一个对象）。扩展名是 `.json` 的 JSONL 也支持。可以粘贴内容，也可以在浏览器选择外部文件；原文件无需放入仓库或修改，只有配置中的样例保存在本地工作区数据库中。

每条样例的三个字段都必须是非空字符串。`instruction` 最多 12000 字符，`input` 和 `output` 各最多 4000 字符。每个场景最多 100 条参考样例、总大小 256 KB。

## 启动

Python 3.11 以上：

```powershell
python -m venv .venv-workbench
.\.venv-workbench\Scripts\python.exe -m pip install -r requirements/intent.txt
.\.venv-workbench\Scripts\python.exe scripts/intent_workbench.py
```

访问 http://127.0.0.1:7861/intent/ 。前端构建文件随源码提供，无需本机模型即可运行配置、生成、审核和导出。数据保存在 `workspace/intent/`。同一工作区只能由一个服务进程使用。

生成配置在 `config/intent_generation.json`，密钥在项目工作目录的 `.env` 或环境变量中：

```dotenv
DEEPSEEK_API_KEY=你的密钥
```

程序不读取 `.env.local`。新建任务时重新读取生成配置；继续旧任务需要保留原模型及参数。`base_url`、`model` 请按实际接口配置。`max_tokens` 默认配置为 8192，以容纳每批完整 JSON 指令；遇到截断时可降低 `batch_size`。

## 按你的 JSON 生成 1000 条

1. 新建项目，例如“北斗大模型训练数据”。
2. 点击“添加场景”：场景名称填写“北斗大模型项目意图识别”，场景标识填写 `dam_intent`。
3. 生成要求填写业务范围、表达风格和答案要求。例如“扩写大坝安全监测问句，覆盖口语与专业表达，依据 instruction 输出全部适用标签”。
4. 在参考样例中粘贴完整 JSONL，或点击“导入 JSON / JSONL 文件”选择你的外部文件。
5. 核对解析条数，展开完整预览检查三个字段。错误会显示具体条目或行号，不能当作普通文本继续保存。
6. 核对“答案标签约束”。页面会尝试从指令中明确列出的标签识别；如适用，应包含完整标签列表，不能只填当前样例出现过的标签。可编辑或清空；非分类任务可以不限制。
7. 计划新增数量填写 **1000**，点击“保存场景”，然后“保存配置”。
8. 进入“样本生成”，点击“按配置开始生成”。每批最多 10 条，最多参考 2 条样例。每次请求独立，不持续追加历史生成内容。1000 条至少需要 100 次请求，过滤重复或无效记录后可能更多。
9. 查看任务进度，支持暂停、继续和停止。失败时保留已完成批次；关闭页面不会停止服务器中的任务。再次点击按配置生成会创建新的新增任务，不是补齐原目标；继续未完成任务请使用“继续”。

**数量含义：新增 1000 条训练记录，参考样例不计入，不自动加入训练集。** 若审核全部通过且没有额外手工样本，则导出为 1000 条。删除或驳回后需按场景补生成差额，再审核；导出页显示实际已通过数量。

## 人工审核

列表显示 `input`、`output`、场景、来源和状态，可展开查看 `instruction`。搜索支持问句、指令和答案。

- 编辑时可修改 instruction、input、output。配置了答案标签约束时，output 用多选框编辑，支持单标签和多标签；否则直接编辑答案字符串。
- 多标签答案按配置顺序用顿号连接，不把标签组合建成新场景。
- 每次编辑、删除恢复后重新进入待审核；原始三个字段与修改历史保留。
- 通过、驳回、删除支持当前页勾选批量处理。切换筛选、页面或版本会清空选择。
- 相同场景下相同 instruction/input 不重复写入，生成也不会复制参考样例。不同场景中相同 instruction/input 对应不同答案会阻止导出，需先处理冲突。
- 模型生成答案仍需要人工核对。格式校验和标签集合校验不能证明业务标注正确。

## 导出与训练

所有待审核项处理完成、没有答案冲突、各场景都有已通过记录后，进入“数据版本”，点击“保存数据版本”。

- “下载 JSON”：直接下载 JSON 数组，只有 `instruction` / `input` / `output` 三个字段。
- “下载训练包 ZIP”：包含 train.json、dataset_info.json、reviewed_samples.json、manifest.json。

新流程保留每条记录的 instruction 和 output，不注入场景 ID，也不改成旧版单标签提示词。冻结版本保存数量和文件校验摘要，后续修改审核记录不会修改已导出的文件。

完整 LlamaFactory WebUI 的“意图分类”页嵌入同一工作台。选择已保存版本及顶部模型，点击“使用内置参数开始微调”，继续由原 Runner 执行。也可将 ZIP 解压后在 Train 页选择数据目录及 `wy_intent_train`。原训练算法、模型加载和评测执行逻辑没有改写。

轻量工作台不加载训练模型。真实微调需要目标机器上的模型、训练依赖和合适的显存。原有评测指标不能替代多标签任务的集合准确率、Micro/Macro-F1；请准备独立测试集。不得把保留用于最终评测的数据同时用作扩写参考样例。

## 旧工作区

数据库进行添加字段的兼容迁移，保留原项目、样本、审核记录和导出文件。旧版文本类别会在场景卡片上标注“旧版文本配置”。编辑并提供完整 JSON 样例、保存配置后创建新版本，不会把旧的单标签数据默默转换成多标签数据。

## 开发与测试

```powershell
.\.venv-intent\Scripts\python.exe -m pip install -r requirements/intent-test.txt
$env:PYTHONPATH = Join-Path (Get-Location) 'src'
$env:WANDB_DISABLED = 'true'
$env:GRADIO_ANALYTICS_ENABLED = 'False'
.\.venv-intent\Scripts\python.exe -m pytest --confcutdir=tests/intent tests/intent -q
npm ci --prefix web/intent
npm run build --prefix web/intent
```

自动测试使用模拟模型/本机模拟接口，不读取真实 API 密钥，不运行微调。JSON 场景测试覆盖格式错误、标签约束、1000 条完整记录、审核修订、删除恢复、下载和兼容迁移。
