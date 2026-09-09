# LlamaFactory-WY

基于 [LlamaFactory](https://github.com/hiyouga/LlamaFactory) 的意图分类微调项目，计划在原有 WebUI 中增加意图配置、数据生成和人工审核，复用已有训练、状态监控、结果展示与基础评测能力。

> 当前状态：已导入上游源码并制定实施计划。下面列出的意图业务功能尚未实现。

## 目标流程

配置意图与样例 → 分批生成分类数据 → 人工查看、修改、删除、审核 → 使用内置参数启动微调 → 查看原有训练与评测结果。

| 环节 | 计划能力 |
| --- | --- |
| 意图配置 | 配置类别标识、名称、说明、样例和各类生成数量 |
| 数据生成 | 调用所选模型，每批参考最多 2 条样例，生成最多 10 条数据；每批独立请求 |
| 数据审核 | 分页筛选、编辑、删除、通过、驳回；修改后重新审核 |
| 微调衔接 | 仅导出审核通过的数据，保存固定版本，自动填入训练参数 |
| 训练与评测 | 复用 LlamaFactory 现有页面和执行流程 |

## 分支与开发文档

- `main`：项目说明，目前只保留此 README。
- [`dev`](https://github.com/XiaoYu1101/LlamaFactory-WY/tree/dev)：完整源码和实施计划，日常开发在此分支进行。
- [实施计划与验收标准](https://github.com/XiaoYu1101/LlamaFactory-WY/blob/dev/docs/WY_IMPLEMENTATION_PLAN.md)：数据模型、生成规则、审核流程、接入位置和分阶段任务。

获取开发代码：

```bash
git clone --branch dev https://github.com/XiaoYu1101/LlamaFactory-WY.git
cd LlamaFactory-WY
```

模型生成使用支持对话与指令的模型。每批结果独立保存，不把全部历史结果追加到上下文；生成数据经人工审核后才能进入该业务流程的训练集。

## 上游来源与许可证

本项目基于 LlamaFactory 源码进行二次开发，导入版本为 `673048c6a543cbbeaed5b8444b8223dc4e23c721`。

开发分支保留上游版权声明和 [Apache-2.0 许可证](https://github.com/XiaoYu1101/LlamaFactory-WY/blob/dev/LICENSE)。上游功能与环境安装说明见开发分支的 [原始英文 README](https://github.com/XiaoYu1101/LlamaFactory-WY/blob/dev/README.upstream.md) 和 [中文 README](https://github.com/XiaoYu1101/LlamaFactory-WY/blob/dev/README_zh.md)。
