# LlamaFactory-WY

基于 [LlamaFactory](https://github.com/hiyouga/LlamaFactory) 的训练数据准备工作台。使用完整的 **instruction / input / output JSON 样例**，按业务场景分批生成训练记录，经人工审核后导出并接入原有微调流程。

## 当前业务流程

配置生成场景 → 粘贴或导入 JSON / JSONL 样例 → 设置新增数量 → 大模型分批扩写 → 查看、编辑、删除、审核 → 下载 JSON 或训练包 → 使用 LlamaFactory 微调。

- 场景标识（例如 `dam_intent`）用于管理任务，不作为模型答案。
- 支持单标签、多标签和其他文本答案；生成和导出保留完整 instruction、input、output。
- 每个场景可配置名称、标识、生成要求、参考 JSON、可选答案标签和新增数量。
- 每批最多参考 2 条样例、生成最多 10 条；支持去重、进度、暂停和继续。
- 参考样例不自动混入训练集。配置新增 1000 条，生成目标就是 1000 条新记录。
- 审核可修改三个字段；多标签答案支持多选。仅导出审核通过的数据，固定版本与后续编辑隔离。
- 训练、状态和原有基础评测继续使用 LlamaFactory；实际模型训练需在目标机器验证。

## 启动

```powershell
pip install -r requirements/intent.txt
python scripts/intent_workbench.py
```

打开 http://127.0.0.1:7861/intent/ 。前端已构建，启动无需 Node.js 或本机模型。

生成接口在 [config/intent_generation.json](config/intent_generation.json) 配置，密钥放入本机 `.env` 或环境变量。原始样例文件可以放在任意目录，从浏览器选择导入即可，无需放入仓库源码。

- [操作指南：使用 JSON 样例生成 1000 条](docs/WY_USAGE.md)
- [本次 JSON 流程改造测试报告](docs/WY_JSON_TEST_REPORT.md)
- [原实施计划](docs/WY_IMPLEMENTATION_PLAN.md)
- [历史测试报告](docs/WY_TEST_REPORT.md)

开发代码在 `dev` 分支。数据工作台按本地单用户、共享工作区设计，运行数据保存在 `workspace/intent/`。

## 上游来源与许可证

项目基于 LlamaFactory 源码二次开发，导入版本为 `673048c6a543cbbeaed5b8444b8223dc4e23c721`，保留上游版权声明及 [Apache-2.0 许可证](LICENSE)。上游安装及训练说明见 [原始 README](README.upstream.md) 和 [中文 README](README_zh.md)。
