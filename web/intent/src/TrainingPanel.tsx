import { useEffect, useState } from "react";
import {
  Alert,
  Button,
  Card,
  Col,
  Form,
  Input,
  InputNumber,
  Row,
  Select,
  Space,
  App,
} from "antd";

type Plan = {
  id: string;
  export_id: string;
  created: string;
  config: Record<string, unknown>;
};
type Options = {
  templates: string[];
  exports: {
    id: string;
    name: string;
    created: string;
    manifest: { count: number; training_ready: boolean };
  }[];
};
async function request<T>(path: string, body?: unknown): Promise<T> {
  const response = await fetch(
    `/intent-api${path}`,
    body === undefined
      ? undefined
      : {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        },
  );
  const data = await response.json();
  if (!response.ok)
    throw new Error(
      typeof data.detail === "string"
        ? data.detail
        : "参数校验失败，请检查填写内容",
    );
  return data;
}
const defaults = {
  lora_rank: 8,
  lora_alpha: null,
  lora_dropout: 0,
  lora_target: "all",
  learning_rate: 0.00005,
  num_train_epochs: 3,
  per_device_train_batch_size: 1,
  gradient_accumulation_steps: 8,
  cutoff_len: 2048,
  precision: "auto",
  val_size: 0,
};
export default function TrainingPanel({
  mode,
  integrated,
}: {
  mode: "train" | "eval";
  integrated: boolean;
}) {
  const training = mode === "train";
  const [form] = Form.useForm();
  const { message } = App.useApp();
  const [options, setOptions] = useState<Options>({
    templates: [],
    exports: [],
  });
  const [plans, setPlans] = useState<Plan[]>([]);
  const [saved, setSaved] = useState<Plan | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  async function refresh() {
    try {
      const [choices, history] = await Promise.all([
        request<Options>("/training/options"),
        request<Plan[]>("/training/configs"),
      ]);
      setOptions(choices);
      setPlans(history);
      setError("");
    } catch (e) {
      setError((e as Error).message);
    }
  }
  useEffect(() => {
    if (training) void refresh();
  }, [training]);
  function openNative() {
    if (window.parent !== window)
      window.parent.postMessage(
        { type: "wy-native-tab", tab: mode },
        window.location.origin,
      );
    else window.location.assign(`/?wy_tab=${mode}`);
  }
  async function save(values: Record<string, unknown>) {
    setBusy(true);
    try {
      const { export_id, ...config } = values;
      const result = await request<Plan>(
        `/exports/${export_id}/training-config`,
        config,
      );
      setSaved(result);
      await refresh();
      message.success("训练配置已保存，可以下载迁移包或填入原生训练页");
    } catch (e) {
      message.error((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  const numeric = [
    ["lora_rank", "LoRA Rank", 1, 256],
    ["lora_alpha", "LoRA Alpha（留空为 Rank × 2）", 1, 2048],
    ["lora_dropout", "LoRA Dropout", 0, 0.99],
    ["learning_rate", "学习率", 0.000000001, 1],
    ["num_train_epochs", "训练轮数", 0.01, 1000],
    ["per_device_train_batch_size", "每设备批大小", 1, 1024],
    ["gradient_accumulation_steps", "梯度累积步数", 1, 4096],
    ["cutoff_len", "最大序列长度", 32, 131072],
    ["val_size", "验证集比例（0 为不划分）", 0, 0.49],
  ] as const;
  return (
    <Card
      title={training ? "模型微调 · LoRA 配置" : "评测与预测 · LlamaFactory"}
    >
      {training ? (
        <>
          <p>
            选择已审核并导出的数据，填写训练机器上的基座模型路径或模型仓库
            ID。这里配置的是本地训练模型，与扩写数据使用的 API 服务分别设置。
          </p>
          {error && <Alert type="error" title={error} />}
          <Space wrap style={{ marginBottom: 16 }}>
            <Button onClick={refresh}>刷新数据与配置</Button>
            <Select
              placeholder="恢复已保存的训练配置"
              style={{ width: 380 }}
              value={saved?.id}
              options={plans.map((p) => ({
                value: p.id,
                label: `${p.config.model_name_or_path} · ${p.created} · ${p.id.slice(0, 8)}`,
              }))}
              onChange={(id) => {
                const plan = plans.find((p) => p.id === id)!;
                form.resetFields();
                form.setFieldsValue({
                  ...plan.config,
                  export_id: plan.export_id,
                });
                setSaved(plan);
              }}
            />
          </Space>
          <Form
            form={form}
            layout="vertical"
            initialValues={defaults}
            onFinish={save}
            onValuesChange={() => setSaved(null)}
            disabled={busy}
          >
            <Form.Item
              name="export_id"
              label="训练数据版本"
              rules={[{ required: true, message: "请选择已导出的数据" }]}
            >
              <Select
                placeholder="先在数据版本页完成审核并导出"
                options={options.exports.map((e) => ({
                  value: e.id,
                  disabled: !e.manifest.training_ready,
                  label: `${e.name} · ${e.manifest.count} 条 · ${e.created} · ${e.id.slice(0, 8)}${e.manifest.training_ready ? "" : "（缺少训练字段映射）"}`,
                }))}
              />
            </Form.Item>
            <Row gutter={20}>
              <Col xs={24} md={12}>
                <Form.Item
                  name="model_name_or_path"
                  label="基座模型路径或 ID"
                  rules={[{ required: true, whitespace: true }]}
                >
                  <Input placeholder="例如 /models/Qwen3-4B 或模型仓库 ID" />
                </Form.Item>
              </Col>
              <Col xs={24} md={12}>
                <Form.Item
                  name="template"
                  label="模型对话模板"
                  extra="必须匹配模型版本；Qwen2/2.5 常用 qwen，Qwen3 非思考任务可选 qwen3_nothink，请按模型说明确认。"
                  rules={[{ required: true }]}
                >
                  <Select
                    showSearch
                    options={options.templates.map((value) => ({
                      value,
                      label: value,
                    }))}
                    placeholder="从当前项目支持的模板中选择"
                  />
                </Form.Item>
              </Col>
              {numeric.map(([name, label, min, max]) => (
                <Col xs={24} md={12} lg={8} key={name}>
                  <Form.Item
                    name={name}
                    label={label}
                    rules={name === "lora_alpha" ? [] : [{ required: true }]}
                  >
                    <InputNumber
                      min={min}
                      max={max}
                      style={{ width: "100%" }}
                    />
                  </Form.Item>
                </Col>
              ))}
              <Col xs={24} md={12}>
                <Form.Item
                  name="lora_target"
                  label="LoRA 目标模块"
                  extra="默认 all；自定义时用英文逗号分隔，例如 q_proj,v_proj。"
                  rules={[{ required: true }]}
                >
                  <Input />
                </Form.Item>
              </Col>
              <Col xs={24} md={12}>
                <Form.Item name="precision" label="计算精度">
                  <Select
                    options={[
                      {
                        value: "auto",
                        label: "自动：按训练机器 CUDA 能力选择",
                      },
                      ...["bf16", "fp16", "fp32"].map((value) => ({
                        value,
                        label: value,
                      })),
                    ]}
                  />
                </Form.Item>
              </Col>
            </Row>
            <p>
              固定使用 SFT + LoRA；不启用量化。其他参数由 LlamaFactory
              默认值和基础运行配置提供。验证集用于观察 loss，分类准确率和 F1
              需独立评测。
            </p>
            <Button type="primary" htmlType="submit" loading={busy}>
              保存训练配置
            </Button>
          </Form>
          {saved && (
            <Alert
              style={{ marginTop: 16 }}
              type="success"
              title={`配置已保存 · ${saved.id.slice(0, 8)}`}
              description={
                <>
                  <p>
                    迁移包包含训练数据、字段映射、train_config.json
                    和启动脚本。保存和下载不会开始训练。
                  </p>
                  <Button
                    href={`/intent-api/training/configs/${saved.id}/download`}
                  >
                    下载 LoRA 训练包
                  </Button>
                  <pre>
                    python run_training.py --dry-run{"\n"}python run_training.py
                    --model /实际模型路径
                  </pre>
                  <p>
                    在目标机器安装本项目及适配硬件的 PyTorch 后运行。输出为 LoRA
                    适配器，使用时仍需基座模型；已有输出不会被覆盖。
                  </p>
                </>
              }
            />
          )}
        </>
      ) : (
        <Alert
          type="info"
          title="使用独立测试数据"
          description="完整界面“意图分类”页下方可选择独立评测数据版本，点击“将数据填入评测页”，自动填入目录和数据集。顶部选择基座模型与 LoRA 适配器，再在原生评测页开始。多标签准确率、Micro/Macro-F1 需要相应任务评测脚本；验证 loss 不代表分类质量。"
        />
      )}
      <div style={{ marginTop: 24 }}>
        {integrated ? (
          <>
            <Button onClick={openNative}>
              {training ? "打开原生训练页" : "打开原生评测页"}
            </Button>
            <p>
              完整界面“意图分类”页下方可刷新并选择已保存的 LoRA
              配置，填入原生训练页；核对模型后点击原生开始按钮，进度、日志和损失曲线沿用
              LlamaFactory。
            </p>
          </>
        ) : (
          <Alert
            type="info"
            title="当前为轻量模式，可保存配置和下载训练包"
            description={
              <>
                <p>也可在训练机器启动完整界面，使用原生训练、评测和监控：</p>
                <pre>
                  python -m pip install -e .{"\n"}python
                  scripts/intent_workbench.py --full
                </pre>
                <p>同一工作区请先关闭轻量服务再切换完整模式。</p>
              </>
            }
          />
        )}
      </div>
    </Card>
  );
}
