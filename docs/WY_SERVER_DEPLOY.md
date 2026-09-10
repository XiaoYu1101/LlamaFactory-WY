# Linux 服务器部署

仓库分支：`dev`。前端生产构建已随源码提交，直接运行不需要安装 Node.js 或重新构建前端。项目要求 Python >= 3.11，以下按 Python 3.11 举例。

## 1. 拉取与安装

首次部署：

```bash
git clone --branch dev https://github.com/XiaoYu1101/LlamaFactory-WY.git
cd LlamaFactory-WY
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

已有克隆时，在项目目录执行 `git pull --ff-only origin dev`；已有本地修改时先自行保留这些修改。训练机需安装与硬件、驱动匹配的 PyTorch。安装项目后检查 CUDA 是否可用：

```bash
nvidia-smi
python -c "import torch; print('torch:', torch.__version__, 'CUDA:', torch.version.cuda, 'available:', torch.cuda.is_available(), 'GPUs:', torch.cuda.device_count())"
```

只有需要 GPU 训练时才要求 CUDA 可用。若不训练、仅扩写和审核，可用 `python -m pip install -r requirements/intent.txt` 代替完整安装，并在启动时省略 `--full`。

## 2. 修改生成服务配置

首次部署复制模板（已有 .env 不要覆盖）：

```bash
cp -n .env.example .env
nano .env
```

最少核对以下配置：

```dotenv
WY_INTENT_BASE_URL=http://127.0.0.1:8000/v1
WY_INTENT_MODEL=auto
WY_INTENT_API_KEY=
WY_INTENT_JSON_MODE=false
WY_INTENT_THINKING=null
WY_INTENT_BATCH_SIZE=10
WY_INTENT_SEED_COUNT=3
```

- BASE_URL 必须是从服务器可访问的 OpenAI 兼容接口地址；127.0.0.1 指当前服务器或当前容器，不是原 Windows 电脑。服务在其他主机时填写其可访问地址。
- MODEL 为 auto 时通过 /models 发现单个模型；返回多个模型时，填写接口返回的准确模型 id。也可从页面“生成服务配置 → 检测服务模型”查看。
- 本机/内网服务无鉴权时 API_KEY 可留空；有鉴权时填写对应密钥。不要上传 .env。
- 默认每批从用户提供的至少 4 条参考中选 3 条，生成 10 条。服务不支持结构化 JSON 参数时保留 JSON_MODE=false。
- 已设置的系统环境变量优先于 .env，.env 优先于 config/intent_generation.json。配置 .env 后通常不需要修改此 JSON。

## 3. 启动

在项目根目录、已激活虚拟环境的终端运行：

```bash
python scripts/intent_workbench.py --full --host 0.0.0.0 --port 7861 --data-dir /srv/llamafactory-wy/intent
```

请把数据目录替换为当前运行账号可写的持久目录。完整界面访问 `http://服务器IP:7861/`，数据工作台访问 `http://服务器IP:7861/intent/`。该服务没有工作台登录鉴权，监听全部网卡时仅向可信内网开放端口。

也可将 host 设为 127.0.0.1，并通过 SSH 转发访问：

```bash
ssh -L 7861:127.0.0.1:7861 用户名@服务器IP
```

浏览器打开本机 `http://127.0.0.1:7861/`。同一个数据目录不能被两个服务同时使用；更新后由你停止旧进程，再使用相同目录启动。

## 4. 训练参数

.env 配置的是生成/推断接口，本地微调参数在“模型训练”页单独设置：

1. 选择已审核导出的数据版本。
2. 将模型路径改为服务器上的模型权重目录，例如 /models/Qwen3-4B，或填写能下载的模型仓库 ID。
3. 选择与模型匹配的模板；不要只根据“Qwen”名称猜测具体版本。常用 LoRA 参数已提供默认值，可以按显存和任务调整。
4. 保存配置。在完整界面顶部先选择模型名称（自定义模型可选 Custom），再到“意图分类”页下方刷新 LoRA 配置并填入原生训练页。
5. 核对后在 Train 启动训练。数据目录、字段映射和 LoRA 参数自动传入；日志、进度、检查点和损失图使用原生功能。

评测需要另行准备独立测试数据。在“意图分类”页下方选择独立评测数据版本，一键填入 Evaluate & Predict；顶部选择正确的基座模型及训练得到的 LoRA 适配器。验证 loss 和原生生成指标不等同于多标签准确率/F1。

## 5. 已有数据怎样搬过去

Git 只包含程序与构建，不包含 .env、模型权重或 workspace 下的本地样本与数据库。服务器首次启动出现空工作区是正常情况。

如果已有审核好的数据，优先从原机器“模型训练”页保存配置并下载 LoRA 训练包，将包复制到服务器解压，激活安装了本项目的环境后执行：

```bash
python /训练包所在目录/run_training.py --dry-run
python /训练包所在目录/run_training.py --model /服务器上的实际模型目录
```

包中数据与输出路径自动按解压目录解析；模板不匹配时修改包内 train_config.json。dry-run 只查看参数，不启动训练。输出目录非空时脚本拒绝覆盖，需要更换 output_dir。

普通数据包中的 train.json 与 dataset_info.json 也可用于原生训练。工作区数据库会记录本机路径，不能假定将 Windows workspace 原样复制到 Linux 就能直接恢复全部导出历史；训练迁移请使用上述训练包。

## 验证范围

当前版本已通过工作台隔离回归、原生训练/评测参数桥接检查和离线浏览器验证；真实 GPU 模型加载、训练与评测仍需在目标服务器验证。

完整模式默认不自动打开服务器上的浏览器。旧版本出现 MoTTY X11、cannot open display 或 Snap 浏览器挂载提示时，可能是自动打开浏览器失败；这些提示本身不说明 Web 服务启动失败。可在服务器另一个终端运行 `curl -I http://127.0.0.1:7861/intent/` 检查 HTTP 响应，并从客户端浏览器访问服务。更新后重启可禁用该自动浏览器行为。
