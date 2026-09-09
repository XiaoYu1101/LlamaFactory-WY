# 意图分类工作台使用说明

## 1. 当前已实现

- 意图配置：项目、类别标识、名称、说明、样例、生成配额；可将总量均分。
- 生成：可替换的 OpenAI 兼容接口配置；最多 2 条种子、每批最多 10 条；独立请求、去重、进度、暂停、停止和中断恢复。
- 审核：筛选、搜索、分页、编辑、人工添加、批量通过/驳回、软删除和恢复；内容修改后重新审核。
- 导出：只导出已审核通过的固定版本，包含 LlamaFactory 可读取的数据和登记文件。
- 完整 WebUI：增加“意图分类”页，提供原训练入口的参数衔接。未运行实际微调和评测，需在你的训练机器上验证。

工作台按单用户、一个共享工作区设计。它没有用户账号及多级审批功能。

## 2. 无需本地模型的启动方式

在仓库根目录、Python 3.11 或以上的环境执行：

```bash
pip install -r requirements/intent.txt
python scripts/intent_workbench.py
```

浏览器访问 `http://127.0.0.1:7861`，自动进入 Ant Design 工作台。该命令只需要 FastAPI、Uvicorn 等轻量依赖，不安装或加载 Torch、Transformers、模型权重。前端构建文件已随仓库提供，使用时无需安装 Node.js。

默认数据库与生成文件放在 `workspace/intent/`。可指定端口、工作区和生成配置：

```bash
python scripts/intent_workbench.py --port 7861 --data-dir workspace/intent --config config/intent_generation.json
```

关闭页面不会停止后台生成。退出服务器进程会中断当前请求；重启并打开原项目、选择任务后点击“继续”，已保存批次不会重做。

同一个数据目录同时只允许一个服务进程使用。切换到完整 WebUI 前先关闭轻量工作台，或者使用另一个数据目录。

## 3. 替换生成配置

修改 `config/intent_generation.json`，无需改业务代码：

| 字段 | 含义 |
| --- | --- |
| provider | `openai_compatible` 使用 API；`local` 使用完整 WebUI 的 Chat 模型 |
| base_url | API 服务地址，例如 `https://api.deepseek.com`，或本机服务 `http://127.0.0.1:8000/v1`；程序追加 `/chat/completions` |
| model | 生成模型名称；本地模式时填写与 Chat 加载模型完全一致的模型路径 |
| api_key_env | 保存密钥的环境变量名，默认 `DEEPSEEK_API_KEY` |
| batch_size | 每批 1 到 10 条，默认 10 |
| seed_count | 每批最多参考 1 到 2 条样例，默认 2；只有一条种子时正常运行 |
| timeout_seconds | 单次网络等待设置，默认 60 秒 |
| max_tokens | 输出 token 上限，默认 2048；截断时可增大，或减小批大小 |
| temperature | 生成采样温度，默认 1.0 |
| json_mode | 是否向接口请求 JSON 对象格式，默认 `true` |
| thinking | DeepSeek 的思考开关，默认 `disabled`；其他不支持此参数的接口设为 `null` |

API 密钥单独写在本机 `.env` 文件中，或由环境变量提供。环境变量优先。

```dotenv
DEEPSEEK_API_KEY=替换为你自己的密钥
```

`.env.example` 为空模板。`.env`、运行数据库、生成数据和日志均被 Git 忽略；仓库中不包含测试密钥。

新建生成任务时重新读取配置文件。继续旧任务时必须保持原来的模型与生成参数一致；允许替换密钥。需要换模型或采样设置时，新建生成任务。

本机 OpenAI 兼容服务可使用如下配置：

```json
{
  "provider": "openai_compatible",
  "base_url": "http://127.0.0.1:8000/v1",
  "model": "your-served-model",
  "api_key_env": "LOCAL_MODEL_API_KEY",
  "timeout_seconds": 60,
  "max_tokens": 2048,
  "temperature": 1.0,
  "json_mode": false,
  "thinking": null,
  "batch_size": 10,
  "seed_count": 2
}
```

如果本机服务不检查密钥，可将 `LOCAL_MODEL_API_KEY` 设置为该服务接受的占位值。提示词仍要求 JSON，客户端始终进行解析和格式校验。

DeepSeek 接口字段参考其[对话接口文档](https://api-docs.deepseek.com/api/create-chat-completion/)和[JSON 输出说明](https://api-docs.deepseek.com/guides/json_mode/)。本次联调使用 `deepseek-v4-flash`，模型可替换。

## 4. 操作步骤

1. 点击“新建项目”，在“意图配置”填写项目名称。点击“添加类别”，在抽屉中填写标识、名称、说明、样例和数量；多条样例按换行分隔。
2. 点击“保存配置”。配置变化会创建新版本；相同配置仅保存项目名称，不重置已有数据。
3. 在“样本生成”点击“按配置开始生成”。例如配额 1000 指新增 1000 条有效待审核数据，原始种子另计。已审核通过的数量单独统计。
4. 查看自动更新的生成进度；点击“审核样本”查看结果。审核页的刷新按钮可读取新数据，轮询进度不覆盖正在编辑的审核列表。
5. 在“数据审核”筛选、搜索和勾选样本，批量通过或驳回。点击每行的编辑图标，在右侧抽屉中修改文本和类别，再点击“保存为待审核”。
6. 删除后可打开“回收站”找到记录并恢复；恢复及编辑都会将状态设为待审核。
7. 全部待审核项处理完成、没有跨标签冲突、各类别都有已通过数据后，在“数据版本”点击“保存数据版本”。
8. 下载 ZIP，或在完整 WebUI 使用该数据版本启动原有微调。

表头复选框和批量操作只作用于当前页明确选中的记录。切换筛选条件、项目或版本会清空选择。列表被其他操作修改后，旧选择会被拒绝，需刷新再处理。页面顶部可切换历史配置，历史样本仍可审核及导出，新生成任务使用当前配置。

补生成按所选类别新增指定数量，不会默默替换被驳回或删除的样本。重复检查包含历史删除/驳回记录，避免反复生成已处理过的同一文本。

## 5. 数据文件与训练衔接

每个固定版本在 `workspace/intent/datasets/<版本 ID>/` 下，包括：

| 文件 | 用途 |
| --- | --- |
| train.json | Alpaca 格式训练数据：instruction、input、output |
| dataset_info.json | 将 `wy_intent_train` 注册到 train.json |
| reviewed_samples.json | 本版本审核样本、来源和样本修订号 |
| manifest.json | 类别定义、统一分类提示词、样本数量和文件校验摘要 |
| dataset.zip | 方便转移到训练机器的压缩包 |

在原 Train 页面将数据目录指向解压目录，选择 `wy_intent_train`，再按原方式训练。采用生成式分类：模型答案是固定类别标识，例如 `refund_status`，不需要新增分类头。

使用 `llamafactory-cli webui` 启动完整 WebUI，其中“意图分类”页嵌入同一 Ant Design 工作台。完成审核后，在工作台下方点击“刷新数据版本”，选择数据版本和顶部的本地训练模型，再点击“使用内置参数开始微调”，由原 Runner 执行，进度、日志、结果仍在 Train 页面。预设为 SFT + LoRA、学习率 5e-5、3 个 epoch、rank 8、batch size 1、梯度累积 8、截断长度 2048；精度依据设备能力选择。模型模板、显存和长样本的配置需在目标机器验证。

按照本次需求，未运行实际训练、模型加载和评测，也未下载任何模型。API 密钥只用于生成样本。

没有自动拆分训练/测试集。请用独立真实数据评测；原有 ROUGE/BLEU 和 token 级 accuracy 不等同于意图分类准确率。

## 6. 测试

```bash
pip install -r requirements/intent-test.txt
```

Linux/macOS：

```bash
PYTHONPATH=src WANDB_DISABLED=true GRADIO_ANALYTICS_ENABLED=False python -m pytest --confcutdir=tests/intent tests/intent -q
```

PowerShell：

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
$env:WANDB_DISABLED = 'true'
$env:GRADIO_ANALYTICS_ENABLED = 'False'
python -m pytest --confcutdir=tests/intent tests/intent -q
```

这组测试使用模拟模型和本机模拟 API，不读取真实密钥、不收费、不运行微调。`--confcutdir` 使轻量测试不加载上游 GPU 训练测试的公共配置。实际 DeepSeek 联调结果另见 `WY_TEST_REPORT.md`。

## 7. 修改 Ant Design 前端

前端源代码位于 `web/intent/`，采用 React、TypeScript 和 Ant Design 6。修改时需要 Node.js 22.12 或以上；普通部署使用仓库中预构建的静态文件即可。

```bash
cd web/intent
npm ci
npm run build
```

构建输出到 `src/llamafactory/intent/static/`，与源码一起提交。开发时启动 Python 工作台，再执行 `npm run dev`，Vite 会将业务接口转发到本机 7861 端口。

独立页面位于 `/intent/`，业务接口位于 `/intent-api/`；完整 WebUI 在相同路径挂载页面和接口，原 Gradio 路由保持可用。数据服务集中在 `src/llamafactory/intent/`，新界面复用同一套存储、生成和审核逻辑。
