# LlamaFactory-WY

基于 [LlamaFactory](https://github.com/hiyouga/LlamaFactory) 的训练数据准备工作台。使用完整的 **JSON 样例（包括 system、自定义字段和对话数组）**，按业务场景分批生成训练记录，经人工审核后导出并接入原有微调流程。

## 当前业务流程

配置生成场景 → 表格填写或导入 JSON / JSONL 样例 → 设置新增数量 → 大模型分批扩写 → 查看、编辑、删除、审核 → 下载 JSON 或训练包 → 使用 LlamaFactory 微调。

- 场景标识（例如 `dam_intent`）用于管理任务，不作为模型答案。
- 不会写 JSON 也可使用表格：选择常用字段模板、逐项填写、自定义字段、增删参考样例；与 JSON 模式互相切换。
- 按参考样例识别字段名、类型和嵌套结构，生成、审核和导出保留完整记录；支持 system、messages、conversations 以及自定义字段。
- 每个场景可配置名称、标识、生成要求、参考 JSON、可选答案标签和新增数量。
- 答案标签由大模型推断，完成后可手动增删。推断任务与开始时的草稿持久保存，关闭页面后从“标签推断任务”恢复；失败或中断支持手动重试。
- 每个场景需要 4–100 条不同的参考样例；默认每批选 3 条生成最多 10 条，优先选择使用较少的样例并尽量避开上一批。支持去重、进度、暂停和继续。
- 参考样例不自动混入训练集。配置新增 1000 条，生成目标就是 1000 条新记录。
- 审核按实际字段提供编辑器；多标签答案支持多选。仅导出审核通过的数据，固定版本与后续编辑隔离。
- 标准问答和对话格式自动映射到 LlamaFactory，自定义文本字段可配置问题、答案、system 映射。无法映射的结构可原样导出 JSON，但不能直接启动训练。
- 数据审核支持跨分页的一键通过、一键审核并下载 JSON；保留已驳回和已删除记录，失败时回滚本次批量审核。
- 左侧提供模型训练、模型评测入口；用 `python scripts/intent_workbench.py --full` 启动完整模式，接入原生训练、评测、Chat 与训练监控。实际模型训练需在目标机器验证。

## 启动

```powershell
pip install -r requirements/intent.txt
python scripts/intent_workbench.py
```

打开 http://127.0.0.1:7861/intent/ 。前端已构建，启动无需 Node.js 或本机模型。

生成服务地址、模型名和密钥优先在本机 `.env` 或环境变量配置，参考 [.env.example](.env.example)；模型名填 `auto` 可发现服务中的单个模型。未设置的选项回退到 [config/intent_generation.json](config/intent_generation.json)。原始样例文件可以放在任意目录，从浏览器选择导入即可，无需放入仓库源码。

- [操作指南：使用 JSON 样例生成 1000 条](docs/WY_USAGE.md)
- [本次 JSON 流程改造测试报告](docs/WY_JSON_TEST_REPORT.md)
- [原实施计划](docs/WY_IMPLEMENTATION_PLAN.md)
- [历史测试报告](docs/WY_TEST_REPORT.md)

开发代码在 `dev` 分支。数据工作台按本地单用户、共享工作区设计，运行数据保存在 `workspace/intent/`。

## 上游来源与许可证

项目基于 LlamaFactory 源码二次开发，导入版本为 `673048c6a543cbbeaed5b8444b8223dc4e23c721`，保留上游版权声明及 [Apache-2.0 许可证](LICENSE)。上游安装及训练说明见 [原始 README](README.upstream.md) 和 [中文 README](README_zh.md)。

LoRA 配置：左侧“模型训练”提供常用参数、配置保存/恢复和训练迁移包下载。完整模式可将保存的配置填入原生 Train；迁移包通过原生 CLI 运行。详见 [LoRA 参数与迁移训练](docs/WY_USAGE.md#lora-参数与迁移训练)。

Linux 服务器部署与迁移配置见 [服务器部署指南](docs/WY_SERVER_DEPLOY.md)。
