import React, { useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  App,
  Alert,
  Button,
  Card,
  ConfigProvider,
  Drawer,
  Empty,
  Form,
  Input,
  InputNumber,
  Menu,
  Modal,
  Popconfirm,
  Progress,
  Select,
  Spin,
  Switch,
  Table,
  Tag,
  Tooltip,
} from "antd";
import zhCN from "antd/locale/zh_CN";
import {
  ApartmentOutlined,
  ArrowRightOutlined,
  CheckCircleOutlined,
  CheckOutlined,
  CloseOutlined,
  CloudServerOutlined,
  DatabaseOutlined,
  DeleteOutlined,
  DownloadOutlined,
  EditOutlined,
  ExperimentOutlined,
  FileDoneOutlined,
  FolderOpenOutlined,
  PauseOutlined,
  PlayCircleOutlined,
  PlusOutlined,
  ReloadOutlined,
  SettingOutlined,
  StopOutlined,
} from "@ant-design/icons";
import type { TableColumnsType } from "antd";
import "./style.css";

type TrainingRecord = { instruction: string; input: string; output: string };
type ParsedExamples = {
  examples: TrainingRecord[];
  count: number;
  output_labels: string[];
};
type Intent = {
  label: string;
  name: string;
  description: string;
  examples: (TrainingRecord | string)[];
  format?: "alpaca" | "text";
  output_labels?: string[];
  target: number;
};
type Project = {
  id: string;
  name: string;
  current_version: string;
  versions: { id: string; created: string }[];
};
type Sample = {
  id: string;
  revision: number;
  label: string;
  text: string;
  original: string;
  instruction: string;
  output: string;
  original_instruction: string;
  original_output: string;
  status: string;
  source: string;
  conflict: boolean;
};
type Job = {
  id: string;
  status: string;
  accepted: number;
  produced: number;
  duplicates: number;
  invalid: number;
  attempts: number;
  targets: Record<string, number>;
  error: string;
  created: string;
};
type Export = {
  id: string;
  path: string;
  created: string;
  manifest: { count: number };
};
type Version = {
  id: string;
  config: Intent[];
  counts: {
    groups: { status: string; deleted: number; n: number; label: string }[];
    conflicts: number;
  };
  jobs: Job[];
  exports: Export[];
};
type Settings = {
  model: string;
  provider: string;
  base_url: string;
  batch_size: number;
  seed_count: number;
  integrated: boolean;
  api_key_env: string;
  api_key_configured: boolean;
  config_path: string;
};
type Page = "config" | "generate" | "review" | "versions";
const states: Record<string, string> = {
  pending: "待审核",
  approved: "已通过",
  rejected: "已驳回",
  queued: "排队中",
  running: "生成中",
  pausing: "正在暂停",
  paused: "已暂停",
  completed: "已完成",
  failed: "失败",
  cancelled: "已停止",
  cancelling: "正在停止",
  interrupted: "已中断",
};
const sources: Record<string, string> = {
  user: "用户样例",
  generated: "模型生成",
  manual: "人工添加",
};
const active = ["queued", "running", "pausing", "cancelling"];
const date = (value: string) =>
  new Date(value).toLocaleString("zh-CN", { hour12: false });
const number = (value: number) => value.toLocaleString("zh-CN");
const nav = [
  { key: "config", icon: <ApartmentOutlined />, label: "场景配置" },
  { key: "generate", icon: <ExperimentOutlined />, label: "样本生成" },
  { key: "review", icon: <FileDoneOutlined />, label: "数据审核" },
  { key: "versions", icon: <DatabaseOutlined />, label: "数据版本" },
];
const copy: Record<Page, [string, string]> = {
  config: [
    "用 JSON 样例扩展训练集",
    "配置生成场景，粘贴完整 JSON 样例，生成并审核 instruction / input / output 训练记录。",
  ],
  generate: [
    "让样例变成数据集",
    "按照参考 JSON 的任务和答案格式扩写，支持单标签、多标签及其他文本答案。",
  ],
  review: [
    "把好每一条数据的质量",
    "查看、修订和审核样本，让训练数据准确表达你的业务意图。",
  ],
  versions: [
    "准备好下一次微调",
    "将审核通过的数据保存为固定版本，交给 LlamaFactory 训练。",
  ],
};

async function api<T>(path: string, body?: unknown): Promise<T> {
  const response = await fetch(
    `/intent-api${path}`,
    body === undefined
      ? {}
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
        : "输入格式不正确，请检查后重试。",
    );
  return data;
}

function useData<T>(path: string | null, tick = 0) {
  const [result, setResult] = useState<{ path: string; data: T } | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  useEffect(() => {
    let current = true;
    setError("");
    if (!path) return;
    setLoading(true);
    api<T>(path)
      .then((data) => {
        if (current) setResult({ path, data });
      })
      .catch((e) => {
        if (current) setError(e.message);
      })
      .finally(() => {
        if (current) setLoading(false);
      });
    return () => {
      current = false;
    };
  }, [path, tick]);
  return {
    data: result?.path === path ? result.data : undefined,
    error,
    loading,
  };
}

function Workbench() {
  const { message, modal } = App.useApp();
  const [page, setPage] = useState<Page>("config");
  const [projectId, setProjectId] = useState("");
  const [versionId, setVersionId] = useState("");
  const [tick, setTick] = useState(0);
  const [poll, setPoll] = useState(0);
  const [busy, setBusy] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [initial, setInitial] = useState(true);
  const [draftName, setDraftName] = useState("");
  const [draft, setDraft] = useState<Intent[]>([]);
  const [dirty, setDirty] = useState(false);
  const [intentEdit, setIntentEdit] = useState<number | null>(null);
  const [intentForm] = Form.useForm();
  const exampleText = Form.useWatch("examples", intentForm);
  const [parsedExamples, setParsedExamples] = useState<ParsedExamples | null>(
    null,
  );
  const [exampleError, setExampleError] = useState("");
  const [parsing, setParsing] = useState(false);
  const [distribution, setDistribution] = useState<number | null>(1000);
  const [extraOpen, setExtraOpen] = useState(false);
  const [extraForm] = Form.useForm();
  const [label, setLabel] = useState("");
  const [status, setStatus] = useState("");
  const [query, setQuery] = useState("");
  const [search, setSearch] = useState("");
  const [deleted, setDeleted] = useState(false);
  const [tablePage, setTablePage] = useState(1);
  const [selected, setSelected] = useState<React.Key[]>([]);
  const [sampleEdit, setSampleEdit] = useState<Sample | "new" | null>(null);
  const [sampleForm] = Form.useForm();
  const sampleLabel = Form.useWatch("label", sampleForm);
  const projects = useData<Project[]>("/projects", tick);
  const project = useData<Project>(
    projectId ? `/projects/${projectId}` : null,
    tick,
  );
  const version = useData<Version>(
    versionId ? `/versions/${versionId}` : null,
    tick + poll,
  );
  const settings = useData<Settings>("/settings", tick);
  const params = new URLSearchParams({
    label,
    status,
    query,
    deleted: String(deleted),
    page: String(tablePage),
  });
  const samples = useData<{ rows: Sample[]; total: number; page: number }>(
    versionId && page === "review"
      ? `/versions/${versionId}/samples?${params}`
      : null,
    tick,
  );
  const refresh = () => setTick((t) => t + 1);
  const intents = version.data?.config || [];
  const sampleScenario = intents.find((x) => x.label === sampleLabel);
  const labelOptions = intents.map((x) => ({ value: x.label, label: x.name }));
  const groups = version.data?.counts.groups.filter((x) => !x.deleted) || [];
  const count = (s?: string) =>
    groups.filter((x) => !s || x.status === s).reduce((n, x) => n + x.n, 0);
  const running =
    version.data?.jobs.some((j) => active.includes(j.status)) || false;
  const history = !!project.data && versionId !== project.data.current_version;
  const errors = [
    ...new Set(
      [
        projects.error,
        project.error,
        version.error,
        samples.error,
        settings.error,
      ].filter(Boolean),
    ),
  ];

  useEffect(() => {
    if (projects.data && initial) {
      if (projects.data.length) setProjectId(projects.data[0].id);
      else setPage("config");
      setInitial(false);
    }
  }, [projects.data, initial]);
  useEffect(() => {
    if (project.data) {
      setVersionId(project.data.current_version);
      setDraftName(project.data.name);
    }
  }, [project.data?.id, project.data?.current_version]);
  useEffect(() => {
    if (version.data) {
      setDraft(version.data.config);
      setDirty(false);
    }
  }, [version.data?.id]);
  useEffect(() => {
    setLabel("");
    setStatus("");
    setQuery("");
    setSearch("");
    setDeleted(false);
    setTablePage(1);
    setSelected([]);
  }, [versionId]);
  useEffect(() => {
    setSelected([]);
  }, [label, status, query, deleted, tablePage, page]);
  useEffect(() => {
    if (!running) return;
    const timer = window.setInterval(() => setPoll((t) => t + 1), 2000);
    return () => window.clearInterval(timer);
  }, [running]);
  useEffect(() => {
    if (!dirty) return;
    const handler = (event: BeforeUnloadEvent) => {
      event.preventDefault();
    };
    window.addEventListener("beforeunload", handler);
    return () => window.removeEventListener("beforeunload", handler);
  }, [dirty]);

  useEffect(() => {
    setParsedExamples(null);
    setExampleError("");
    if (intentEdit === null || !exampleText?.trim()) {
      setParsing(false);
      return;
    }
    let current = true;
    setParsing(true);
    const timer = window.setTimeout(() => {
      api<ParsedExamples>("/examples/parse", { examples: exampleText })
        .then((result) => {
          if (!current) return;
          setParsedExamples(result);
          if (intentForm.getFieldValue("output_labels") === undefined)
            intentForm.setFieldValue("output_labels", result.output_labels);
        })
        .catch((e) => {
          if (current) setExampleError(e.message);
        })
        .finally(() => {
          if (current) setParsing(false);
        });
    }, 350);
    return () => {
      current = false;
      window.clearTimeout(timer);
    };
  }, [exampleText, intentEdit]);

  function switchContext(action: () => void) {
    if (dirty)
      modal.confirm({
        title: "有未保存的配置",
        content: "切换后会放弃本次修改。",
        okText: "放弃并切换",
        onOk: action,
      });
    else action();
  }
  async function perform(action: () => Promise<unknown>, success: string) {
    setBusy(true);
    try {
      await action();
      message.success(success);
      setSelected([]);
      refresh();
    } catch (e) {
      message.error((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  function newProject() {
    switchContext(() => {
      setProjectId("");
      setVersionId("");
      setDraftName("");
      setDraft([]);
      setDirty(false);
      setPage("config");
    });
  }
  function openIntent(index: number) {
    setIntentEdit(index);
    const item = draft[index];
    intentForm.resetFields();
    intentForm.setFieldsValue(
      item
        ? {
            ...item,
            examples: item.examples
              .map((x) => (typeof x === "string" ? x : JSON.stringify(x)))
              .join("\n"),
          }
        : { label: "", name: "", description: "", examples: "", target: 1000 },
    );
  }
  async function saveProject() {
    await perform(async () => {
      const result = await api<{ id: string; version_id: string }>(
        "/projects",
        { name: draftName, intents: draft, project_id: projectId || null },
      );
      setDirty(false);
      setProjectId(result.id);
      setVersionId(result.version_id);
    }, "配置已保存，可以开始生成样本");
  }
  function editSample(sample: Sample | "new") {
    setSampleEdit(sample);
    sampleForm.resetFields();
    const scenario = intents.find((x) => x.label === label) || intents[0];
    const reference = scenario?.examples[0];
    sampleForm.setFieldsValue(
      sample === "new"
        ? {
            label: scenario?.label,
            text: "",
            instruction:
              typeof reference === "object" ? reference.instruction : "",
            output: "",
          }
        : sample,
    );
  }
  function review(action: string, rows?: Sample[]) {
    const chosen =
      rows ||
      samples.data?.rows.filter((row) => selected.includes(row.id)) ||
      [];
    return perform(
      () =>
        api(`/versions/${versionId}/review`, {
          action,
          selected: chosen.map(({ id, revision }) => ({ id, revision })),
        }),
      "样本状态已更新",
    );
  }
  const columns: TableColumnsType<Sample> = [
    {
      title: "问句 / input",
      dataIndex: "text",
      render: (text, row) => (
        <div className="sample-text">
          <div>{text}</div>
          {row.instruction && (
            <details className="instruction-detail">
              <summary>查看 instruction</summary>
              <p>{row.instruction}</p>
            </details>
          )}
          {row.conflict && <Tag color="error">答案冲突</Tag>}
        </div>
      ),
    },
    {
      title: "答案 / output",
      dataIndex: "output",
      width: 220,
      render: (value, row) =>
        value ? (
          <span className="answer-text">{value}</span>
        ) : (
          <Tag>{row.label}（旧版）</Tag>
        ),
    },
    {
      title: "生成场景",
      dataIndex: "label",
      width: 150,
      render: (value) => (
        <Tag className="intent-tag">
          {intents.find((x) => x.label === value)?.name || value}
        </Tag>
      ),
    },
    {
      title: "状态",
      dataIndex: "status",
      width: 110,
      render: (value) => (
        <span className={`status ${value}`}>
          <i />
          {states[value]}
        </span>
      ),
    },
    {
      title: "来源",
      dataIndex: "source",
      width: 106,
      render: (value) => <span className="muted">{sources[value]}</span>,
    },
    {
      title: "操作",
      key: "actions",
      width: 138,
      fixed: "right",
      render: (_, row) =>
        deleted ? (
          <Button
            type="link"
            disabled={busy}
            onClick={() => review("restore", [row])}
          >
            恢复
          </Button>
        ) : (
          <div className="row-actions">
            <Tooltip title="编辑样本">
              <Button
                type="text"
                aria-label="编辑样本"
                icon={<EditOutlined />}
                disabled={busy}
                onClick={() => editSample(row)}
              />
            </Tooltip>
            <Tooltip title="通过">
              <Button
                type="text"
                aria-label="通过样本"
                icon={<CheckOutlined />}
                disabled={busy || row.status === "approved"}
                onClick={() => review("approved", [row])}
              />
            </Tooltip>
            <Popconfirm
              title="删除这条样本？"
              description="删除后可在回收站恢复。"
              onConfirm={() => review("delete", [row])}
            >
              <Button
                type="text"
                aria-label="删除样本"
                icon={<DeleteOutlined />}
                disabled={busy}
              />
            </Popconfirm>
          </div>
        ),
    },
  ];

  return (
    <div className="workbench">
      <aside className="sidebar">
        <a className="brand" href="/intent/">
          <span className="brand-mark">
            W<span>Y</span>
          </span>
          <div>
            意图工作台<small>LLAMAFACTORY WY</small>
          </div>
        </a>
        <div className="nav-caption">数据工作流</div>
        <Menu
          mode="inline"
          selectedKeys={[page]}
          items={nav}
          onClick={({ key }) => setPage(key as Page)}
        />
        <div className="sidebar-guide">
          <div className="guide-icon">
            <ApartmentOutlined />
          </div>
          <strong>从意图到训练数据</strong>
          <p>
            配置场景，扩展样本，
            <br />
            让每一次微调有据可依。
          </p>
          <div className="guide-steps">
            <span>配置</span>
            <i />
            <span>生成</span>
            <i />
            <span>审核</span>
          </div>
        </div>
        <button
          className="sidebar-settings"
          onClick={() => setSettingsOpen(true)}
        >
          <SettingOutlined /> 生成服务配置
        </button>
        <div className="sidebar-foot">
          <span className="online-dot" /> 本地工作区 <span>v1.0</span>
        </div>
      </aside>
      <div className="main-shell">
        <header className="topbar">
          <div className="workspace-label">
            <FolderOpenOutlined />
            <span>我的工作区</span>
            <b>/</b>
            <Select
              aria-label="切换项目"
              className="project-select"
              variant="borderless"
              placeholder="选择项目"
              value={projectId || undefined}
              disabled={busy}
              options={projects.data?.map((x) => ({
                value: x.id,
                label: x.name,
              }))}
              onChange={(id) =>
                switchContext(() => {
                  setVersionId("");
                  setProjectId(id);
                  setDirty(false);
                })
              }
            />
          </div>
          <div className="topbar-actions">
            <span className="model-indicator">
              <CloudServerOutlined />
              {settings.data?.model || "读取配置中"}
            </span>
            <Button
              icon={<PlusOutlined />}
              onClick={newProject}
              disabled={busy}
            >
              新建项目
            </Button>
          </div>
        </header>
        <main className="content">
          <div className="page-heading">
            <div>
              <div className="eyebrow">
                INTENT WORKSPACE{" "}
                <span>/ {nav.find((x) => x.key === page)?.label}</span>
              </div>
              <h1>{copy[page][0]}</h1>
              <p>{copy[page][1]}</p>
            </div>
            <div className="heading-actions">
              {versionId && (
                <Select
                  aria-label="配置版本"
                  className="version-select"
                  value={versionId}
                  disabled={busy}
                  options={project.data?.versions.map((v, i, all) => ({
                    value: v.id,
                    label: `配置 V${all.length - i}${v.id === project.data?.current_version ? " · 当前" : " · 历史"}`,
                  }))}
                  onChange={(id) =>
                    switchContext(() => {
                      setVersionId(id);
                      setDirty(false);
                    })
                  }
                />
              )}
              <Tooltip title="刷新数据">
                <Button
                  aria-label="刷新数据"
                  icon={<ReloadOutlined />}
                  onClick={() => {
                    setSelected([]);
                    refresh();
                  }}
                  disabled={busy}
                />
              </Tooltip>
            </div>
          </div>
          {errors.map((error) => (
            <Alert
              key={error}
              className="notice"
              type="error"
              showIcon
              title={error}
              action={
                <Button size="small" onClick={refresh}>
                  重试
                </Button>
              }
            />
          ))}
          {history && (
            <Alert
              className="notice"
              type="info"
              showIcon
              title="正在查看历史配置版本"
              description="可以继续审核和导出历史样本。新增生成任务请切回当前配置。"
            />
          )}
          {page === "config" &&
          projectId &&
          (!project.data || !version.data) ? (
            <div className="loading">
              <Spin size="large" />
            </div>
          ) : page === "config" ? (
            <>
              <Card className="config-meta">
                <div className="config-meta-grid">
                  <div>
                    <label className="field-label" htmlFor="project-name">
                      项目名称
                    </label>
                    <Input
                      id="project-name"
                      placeholder="例如：北斗大模型训练数据"
                      maxLength={100}
                      value={draftName}
                      onChange={(e) => {
                        setDraftName(e.target.value);
                        setDirty(true);
                      }}
                      disabled={history || busy}
                    />
                  </div>
                  <div className="config-summary">
                    <span>
                      <strong>{draft.length}</strong> 个场景
                    </span>
                    <span>
                      <strong>
                        {number(draft.reduce((n, x) => n + x.target, 0))}
                      </strong>{" "}
                      条计划生成
                    </span>
                  </div>
                  <Button
                    type="primary"
                    icon={<CheckOutlined />}
                    loading={busy}
                    disabled={history || !draft.length || !draftName.trim()}
                    onClick={saveProject}
                  >
                    保存配置{dirty ? " *" : ""}
                  </Button>
                </div>
              </Card>
              <div className="section-heading">
                <div>
                  <h2>
                    生成场景 <span>{draft.length}</span>
                  </h2>
                  <p>
                    每个场景粘贴 JSONL 或 JSON
                    数组；参考样例仅指导生成，不计入新增数量和训练集。
                  </p>
                </div>
                <Button
                  icon={<PlusOutlined />}
                  disabled={history || busy}
                  onClick={() => openIntent(draft.length)}
                >
                  添加场景
                </Button>
              </div>
              {!draft.length ? (
                <Card>
                  <Empty description="从第一个生成场景开始">
                    <Button
                      type="primary"
                      icon={<PlusOutlined />}
                      onClick={() => openIntent(0)}
                    >
                      添加生成场景
                    </Button>
                  </Empty>
                </Card>
              ) : (
                <div className="intent-grid">
                  {draft.map((intent, index) => (
                    <Card key={index} className="intent-card">
                      <div className="intent-card-head">
                        <div className={`category-symbol color-${index % 4}`}>
                          <ApartmentOutlined />
                        </div>
                        <div>
                          <h3>{intent.name}</h3>
                          <code>{intent.label}</code>
                        </div>
                        <div className="card-actions">
                          <Button
                            type="text"
                            aria-label={`编辑${intent.name}`}
                            icon={<EditOutlined />}
                            disabled={history || busy}
                            onClick={() => openIntent(index)}
                          />
                          <Popconfirm
                            title="从配置中移除此场景？"
                            onConfirm={() => {
                              setDraft(draft.filter((_, i) => i !== index));
                              setDirty(true);
                            }}
                          >
                            <Button
                              type="text"
                              aria-label={`移除${intent.name}`}
                              icon={<DeleteOutlined />}
                              disabled={history || busy}
                            />
                          </Popconfirm>
                        </div>
                      </div>
                      <p className="intent-description">{intent.description}</p>
                      {intent.format !== "alpaca" && (
                        <Tag color="warning">
                          旧版文本配置：请编辑并粘贴 JSON 样例
                        </Tag>
                      )}
                      <div className="examples-label">
                        参考样例 <span>{intent.examples.length}</span>
                      </div>
                      <div className="seed-preview">
                        {intent.examples.slice(0, 2).map((example, i) => (
                          <p key={i}>
                            <span>0{i + 1}</span>
                            {typeof example === "string" ? (
                              example
                            ) : (
                              <span className="json-preview">
                                <strong>input</strong> {example.input}
                                <br />
                                <strong>output</strong> {example.output}
                              </span>
                            )}
                          </p>
                        ))}
                      </div>
                      <div className="intent-card-foot">
                        <span>计划新增样本</span>
                        <strong>
                          {number(intent.target)} <small>条</small>
                        </strong>
                      </div>
                    </Card>
                  ))}
                </div>
              )}
              {!!draft.length && (
                <div className="distribution">
                  <span>按总量分配生成配额</span>
                  <InputNumber
                    aria-label="生成总量"
                    min={draft.length}
                    max={100000}
                    precision={0}
                    value={distribution}
                    onChange={setDistribution}
                    disabled={history || busy}
                  />
                  <Button
                    disabled={
                      history ||
                      busy ||
                      !distribution ||
                      distribution < draft.length
                    }
                    onClick={() => {
                      const total = distribution!;
                      setDraft(
                        draft.map((x, i) => ({
                          ...x,
                          target:
                            Math.floor(total / draft.length) +
                            (i < total % draft.length ? 1 : 0),
                        })),
                      );
                      setDirty(true);
                    }}
                  >
                    均分到各场景
                  </Button>
                  <span className="muted">保存后生效</span>
                </div>
              )}
              {versionId && (
                <div className="next-step">
                  <span>
                    <CheckCircleOutlined /> 配置保存后，即可开始生成训练样本
                  </span>
                  <Button type="link" onClick={() => setPage("generate")}>
                    前往样本生成 <ArrowRightOutlined />
                  </Button>
                </div>
              )}
            </>
          ) : !versionId ? (
            <Card>
              <Empty description="先创建项目并保存场景配置">
                <Button type="primary" onClick={() => setPage("config")}>
                  配置生成场景
                </Button>
              </Empty>
            </Card>
          ) : !version.data ? (
            <div className="loading">
              <Spin size="large" />
            </div>
          ) : (
            <>
              {page === "review" && (
                <>
                  <div className="stats-grid">
                    {[
                      [
                        "样本总量",
                        count(),
                        "生成和人工添加的数据；JSON 参考样例另计",
                        <DatabaseOutlined />,
                      ],
                      [
                        "待审核",
                        count("pending"),
                        "等待人工确认",
                        <FileDoneOutlined />,
                      ],
                      [
                        "已通过",
                        count("approved"),
                        "可用于训练的优质样本",
                        <CheckCircleOutlined />,
                      ],
                      [
                        "答案冲突",
                        version.data.counts.conflicts,
                        "相同指令和问句对应不同答案",
                        <ApartmentOutlined />,
                      ],
                    ].map(([title, value, desc, icon], index) => (
                      <div
                        className={`stat-card stat-${index}`}
                        key={String(title)}
                      >
                        <div className="stat-title">
                          {title}
                          <span>{icon}</span>
                        </div>
                        <div className="stat-number">
                          {number(value as number)}
                          <small>{index === 3 ? "组" : "条"}</small>
                        </div>
                        <div className="stat-description">{desc}</div>
                      </div>
                    ))}
                  </div>
                  <Card className="review-card">
                    <div className="table-heading">
                      <div>
                        <h2>
                          样本库 <Tag>{deleted ? "回收站" : "全部数据"}</Tag>
                        </h2>
                        <p>
                          问句、指令或答案修改后重新审核；多标签答案保存在同一条记录中。
                        </p>
                      </div>
                      <Button
                        icon={<PlusOutlined />}
                        onClick={() => editSample("new")}
                        disabled={busy}
                      >
                        人工添加
                      </Button>
                    </div>
                    <div className="filters">
                      <Input.Search
                        aria-label="搜索问句、指令或答案"
                        placeholder="搜索问句、指令或答案…"
                        value={search}
                        onChange={(e) => {
                          setSearch(e.target.value);
                          if (!e.target.value) {
                            setQuery("");
                            setTablePage(1);
                          }
                        }}
                        onSearch={(value) => {
                          setQuery(value);
                          setTablePage(1);
                        }}
                        allowClear
                        className="sample-search"
                      />
                      <Select
                        aria-label="筛选生成场景"
                        value={label}
                        options={[
                          { value: "", label: "全部生成场景" },
                          ...labelOptions,
                        ]}
                        onChange={(v) => {
                          setLabel(v);
                          setTablePage(1);
                        }}
                      />
                      <Select
                        aria-label="筛选审核状态"
                        value={status}
                        options={[
                          { value: "", label: "全部审核状态" },
                          ...["pending", "approved", "rejected"].map((v) => ({
                            value: v,
                            label: states[v],
                          })),
                        ]}
                        onChange={(v) => {
                          setStatus(v);
                          setTablePage(1);
                        }}
                      />
                      <label className="trash-switch">
                        <Switch
                          size="small"
                          checked={deleted}
                          onChange={(v) => {
                            setDeleted(v);
                            setTablePage(1);
                          }}
                        />{" "}
                        回收站
                      </label>
                    </div>
                    <div
                      className={`selection-bar ${selected.length ? "has-selection" : ""}`}
                    >
                      <span>
                        已选择 <strong>{selected.length}</strong> 条
                        {selected.length ? "样本" : " · 勾选表格开始批量审核"}
                      </span>
                      <div>
                        {deleted ? (
                          <Button
                            size="small"
                            disabled={!selected.length || busy}
                            onClick={() => review("restore")}
                          >
                            恢复选中
                          </Button>
                        ) : (
                          <>
                            <Button
                              size="small"
                              type="primary"
                              icon={<CheckOutlined />}
                              disabled={!selected.length || busy}
                              onClick={() => review("approved")}
                            >
                              通过
                            </Button>
                            <Button
                              size="small"
                              icon={<CloseOutlined />}
                              disabled={!selected.length || busy}
                              onClick={() => review("rejected")}
                            >
                              驳回
                            </Button>
                            <Popconfirm
                              title={`删除选中的 ${selected.length} 条样本？`}
                              onConfirm={() => review("delete")}
                            >
                              <Button
                                size="small"
                                danger
                                icon={<DeleteOutlined />}
                                disabled={!selected.length || busy}
                              >
                                删除
                              </Button>
                            </Popconfirm>
                          </>
                        )}
                      </div>
                    </div>
                    <Table<Sample>
                      rowKey="id"
                      columns={columns}
                      dataSource={samples.data?.rows || []}
                      loading={samples.loading || busy}
                      rowSelection={{
                        selectedRowKeys: selected,
                        onChange: setSelected,
                        getCheckboxProps: () => ({ disabled: busy }),
                      }}
                      pagination={{
                        current: samples.data?.page || tablePage,
                        total: samples.data?.total || 0,
                        pageSize: 25,
                        showSizeChanger: false,
                        showTotal: (total) => `共 ${number(total)} 条样本`,
                        onChange: (p) => setTablePage(p),
                      }}
                      scroll={{ x: 850 }}
                      locale={{
                        emptyText: <Empty description="没有符合条件的样本" />,
                      }}
                    />
                  </Card>
                  <div className="next-step">
                    <span>
                      <FileDoneOutlined /> 处理完全部待审核样本后，保存数据版本
                    </span>
                    <Button type="link" onClick={() => setPage("versions")}>
                      前往数据版本 <ArrowRightOutlined />
                    </Button>
                  </div>
                </>
              )}
              {page === "generate" && (
                <>
                  {settings.data?.provider === "openai_compatible" &&
                    !settings.data.api_key_configured && (
                      <Alert
                        className="notice"
                        type="warning"
                        showIcon
                        title="尚未配置生成 API 密钥"
                        description={`请在项目工作目录的 .env 或环境变量中设置 ${settings.data.api_key_env}，然后刷新页面。模型名称仅表示当前配置，不代表接口已连通。`}
                      />
                    )}
                  <Card className="generation-hero">
                    <div>
                      <Tag color="blue">按场景配置生成</Tag>
                      <h2>少量样例，更多业务表达。</h2>
                      <p>
                        根据已保存的 JSON
                        样例生成完整训练记录。只计算有效新增记录；重复和无效结果会过滤，参考样例不混入训练集。
                      </p>
                      <div className="hero-actions">
                        <Button
                          type="primary"
                          size="large"
                          icon={<PlayCircleOutlined />}
                          loading={busy}
                          disabled={history || running || dirty}
                          onClick={() =>
                            perform(
                              () => api(`/versions/${versionId}/jobs`, {}),
                              "生成任务已启动",
                            )
                          }
                        >
                          按配置开始生成
                        </Button>
                        <Button
                          size="large"
                          disabled={history || running || busy || dirty}
                          onClick={() => {
                            extraForm.setFieldsValue({
                              label: intents[0]?.label,
                              amount: 10,
                            });
                            setExtraOpen(true);
                          }}
                        >
                          按场景补生成
                        </Button>
                      </div>
                      {dirty && (
                        <p className="dirty-hint">
                          配置有未保存修改，请先保存。
                        </p>
                      )}
                    </div>
                  </Card>
                  <div className="generation-meta">
                    <span>
                      <CloudServerOutlined /> {settings.data?.model}
                    </span>
                    <span>{intents.length} 个生成场景</span>
                    <span>
                      本次配置计划新增{" "}
                      <strong>
                        {number(intents.reduce((n, x) => n + x.target, 0))}
                      </strong>{" "}
                      条
                    </span>
                    <Button type="link" onClick={() => setSettingsOpen(true)}>
                      查看生成配置
                    </Button>
                  </div>
                  <div className="section-heading">
                    <div>
                      <h2>生成任务</h2>
                      <p>可暂停和继续。关闭页面后，服务中的任务仍会继续。</p>
                    </div>
                    <Tag>{version.data.jobs.length} 个任务</Tag>
                  </div>
                  {!version.data.jobs.length ? (
                    <Card>
                      <Empty description="暂无生成任务，准备好后开始第一次生成" />
                    </Card>
                  ) : (
                    version.data.jobs.map((job) => {
                      const target = Object.values(job.targets).reduce(
                        (a, b) => a + b,
                        0,
                      );
                      return (
                        <Card className="job-card" key={job.id}>
                          <div className="job-heading">
                            <div>
                              <h3>
                                样本生成 <code>#{job.id.slice(0, 8)}</code>
                              </h3>
                              <span className="muted">
                                {date(job.created)} ·{" "}
                                {Object.keys(job.targets).length} 个场景
                              </span>
                            </div>
                            <Tag
                              color={
                                job.status === "completed"
                                  ? "success"
                                  : job.status === "failed"
                                    ? "error"
                                    : "processing"
                              }
                            >
                              {states[job.status]}
                            </Tag>
                          </div>
                          <Progress
                            percent={Math.min(
                              100,
                              Math.round((job.accepted / target) * 100),
                            )}
                            status={
                              job.status === "failed"
                                ? "exception"
                                : job.status === "completed"
                                  ? "success"
                                  : active.includes(job.status)
                                    ? "active"
                                    : "normal"
                            }
                          />
                          <div className="job-foot">
                            <div>
                              <strong>{number(job.accepted)}</strong> /{" "}
                              {number(target)} 条{" "}
                              <span>
                                已过滤 {job.duplicates + job.invalid}{" "}
                                条重复或无效样本
                              </span>
                            </div>
                            <div className="row-actions">
                              {["running", "queued"].includes(job.status) && (
                                <Button
                                  icon={<PauseOutlined />}
                                  disabled={busy}
                                  onClick={() =>
                                    perform(
                                      () => api(`/jobs/${job.id}/pause`, {}),
                                      "正在暂停，请稍候",
                                    )
                                  }
                                >
                                  暂停
                                </Button>
                              )}
                              {["paused", "interrupted", "failed"].includes(
                                job.status,
                              ) && (
                                <Button
                                  icon={<PlayCircleOutlined />}
                                  disabled={busy || history}
                                  onClick={() =>
                                    perform(
                                      () => api(`/jobs/${job.id}/resume`, {}),
                                      "任务已继续",
                                    )
                                  }
                                >
                                  继续
                                </Button>
                              )}
                              {!["completed", "cancelled"].includes(
                                job.status,
                              ) && (
                                <Button
                                  icon={<StopOutlined />}
                                  disabled={busy}
                                  onClick={() =>
                                    perform(
                                      () => api(`/jobs/${job.id}/cancel`, {}),
                                      "已请求停止",
                                    )
                                  }
                                >
                                  停止
                                </Button>
                              )}
                              <Button
                                type="link"
                                onClick={() => {
                                  setPage("review");
                                  refresh();
                                }}
                              >
                                审核样本 <ArrowRightOutlined />
                              </Button>
                            </div>
                          </div>
                          {job.error && (
                            <Alert type="warning" showIcon title={job.error} />
                          )}
                        </Card>
                      );
                    })
                  )}
                </>
              )}
              {page === "versions" && (
                <>
                  <Card className="export-hero">
                    <div className="export-icon">
                      <DatabaseOutlined />
                    </div>
                    <div>
                      <h2>保存一份确定的训练数据</h2>
                      <p>
                        仅纳入已通过的训练记录，保留
                        instruction、input、output。参考 JSON
                        不自动加入；删除或驳回后，可补生成到目标数量。
                      </p>
                      <div className="export-checks">
                        <Tag color={count("pending") ? "orange" : "success"}>
                          {count("pending")} 条待审核
                        </Tag>
                        <Tag
                          color={
                            version.data.counts.conflicts ? "error" : "success"
                          }
                        >
                          {version.data.counts.conflicts} 组冲突
                        </Tag>
                        <Tag>{count("approved")} 条已通过</Tag>
                      </div>
                    </div>
                    <Button
                      type="primary"
                      size="large"
                      icon={<FileDoneOutlined />}
                      loading={busy}
                      disabled={
                        running ||
                        !!count("pending") ||
                        !!version.data.counts.conflicts ||
                        !count("approved")
                      }
                      onClick={() =>
                        perform(
                          () => api(`/versions/${versionId}/exports`, {}),
                          "数据版本已保存",
                        )
                      }
                    >
                      保存数据版本
                    </Button>
                  </Card>
                  <div className="section-heading">
                    <div>
                      <h2>已保存版本</h2>
                      <p>
                        下载包包含训练数据、数据注册文件、审核样本和版本清单。
                      </p>
                    </div>
                  </div>
                  {!version.data.exports.length ? (
                    <Card>
                      <Empty description="还没有保存的数据版本" />
                    </Card>
                  ) : (
                    version.data.exports.map((item, index, all) => (
                      <Card className="export-card" key={item.id}>
                        <div className="export-row">
                          <div className="file-symbol">
                            <FileDoneOutlined />
                          </div>
                          <div className="export-detail">
                            <h3>
                              数据集 V{all.length - index}{" "}
                              <Tag color="success">已冻结</Tag>
                            </h3>
                            <p>
                              {date(item.created)} ·{" "}
                              {number(item.manifest.count)} 条样本 · Alpaca 格式
                            </p>
                            <code>{item.id.slice(0, 8)}</code>
                          </div>
                          <Button
                            icon={<DownloadOutlined />}
                            href={`/intent-api/exports/${item.id}/download`}
                          >
                            下载训练包 ZIP
                          </Button>
                          <Button
                            icon={<DownloadOutlined />}
                            href={`/intent-api/exports/${item.id}/json`}
                          >
                            下载 JSON
                          </Button>
                        </div>
                        <div className="dataset-path">
                          <span>训练数据目录</span>
                          <code>{item.path}</code>
                        </div>
                      </Card>
                    ))
                  )}
                  <Alert
                    className="training-note"
                    type="info"
                    showIcon
                    title={
                      settings.data?.integrated
                        ? "使用 LlamaFactory 原有训练流程"
                        : "在模型所在机器上启动微调"
                    }
                    description={
                      settings.data?.integrated
                        ? "在本页下方选择已保存的数据版本，使用顶部选择的模型启动微调。训练状态、结果和评测仍使用原有页面。"
                        : "将数据包下载到训练机器并解压，在完整 WebUI 的 Train 页面选择这个数据目录和 wy_intent_train。也可在完整 WebUI 的“意图分类”页直接启动默认参数微调。"
                    }
                  />
                </>
              )}
            </>
          )}
          <footer className="page-footer">
            LlamaFactory WY <span>让数据准备更有条理</span>
          </footer>
        </main>
      </div>
      <Drawer
        title={
          intentEdit !== null && draft[intentEdit]
            ? "编辑生成场景"
            : "添加生成场景"
        }
        size={760}
        open={intentEdit !== null}
        onClose={() => setIntentEdit(null)}
        footer={
          <div className="drawer-footer">
            <Button onClick={() => setIntentEdit(null)}>取消</Button>
            <Button
              type="primary"
              disabled={parsing || !!exampleError || !parsedExamples}
              onClick={() => intentForm.submit()}
            >
              保存场景
            </Button>
          </div>
        }
      >
        <Form
          name="intent-config"
          form={intentForm}
          layout="vertical"
          onFinish={async (values) => {
            try {
              const result = await api<ParsedExamples>("/examples/parse", {
                examples: values.examples,
              });
              const item: Intent = {
                ...values,
                format: "alpaca",
                examples: result.examples,
                output_labels: values.output_labels ?? result.output_labels,
              };
              const validated = await api<Intent>("/scenarios/validate", item);
              const next = [...draft];
              next[intentEdit!] = validated;
              setDraft(next);
              setDirty(true);
              setIntentEdit(null);
            } catch (e) {
              message.error((e as Error).message);
            }
          }}
        >
          <Form.Item
            name="name"
            label="场景名称"
            rules={[
              { required: true, whitespace: true, message: "请输入场景名称" },
            ]}
          >
            <Input placeholder="例如：北斗大模型项目意图识别" maxLength={100} />
          </Form.Item>
          <Form.Item
            name="label"
            label="场景标识"
            extra="用于区分生成场景，例如 dam_intent；训练答案来自 JSON 的 output。"
            rules={[
              {
                required: true,
                pattern: /^[a-zA-Z][a-zA-Z0-9_-]{0,63}$/,
                message:
                  "以字母开头，使用字母、数字、下划线或短横线，最长 64 位",
              },
            ]}
          >
            <Input placeholder="例如：dam_intent" />
          </Form.Item>
          <Form.Item
            name="description"
            label="生成要求"
            rules={[
              {
                required: true,
                whitespace: true,
                message: "请描述期望生成的数据",
              },
            ]}
          >
            <Input.TextArea
              rows={3}
              maxLength={1000}
              placeholder="说明哪些表达属于这个场景，以及与其他类别的边界。"
            />
          </Form.Item>
          <label className="json-file-import">
            导入 JSON / JSONL 文件
            <input
              aria-label="导入 JSON 样例文件"
              type="file"
              accept=".json,.jsonl,application/json,text/plain"
              onChange={async (event) => {
                const file = event.target.files?.[0];
                if (!file) return;
                if (file.size > 256000) {
                  message.error("参考样例文件不能超过 256 KB");
                  return;
                }
                intentForm.setFieldValue("examples", await file.text());
                intentForm.setFieldValue("output_labels", undefined);
                event.target.value = "";
              }}
            />
          </label>
          <Form.Item
            name="examples"
            label="参考样例"
            extra="支持每行一个 JSON 对象（JSONL）或 JSON 数组。每条必须包含 instruction、input、output。"
            rules={[
              {
                required: true,
                whitespace: true,
                message: "请至少提供一条样例",
              },
            ]}
          >
            <Input.TextArea
              rows={10}
              className="json-editor"
              placeholder='{"instruction":"你的任务指令","input":"参考问句","output":"正确答案"}'
            />
          </Form.Item>
          {parsing && (
            <p>
              <Spin size="small" /> 正在解析 JSON 样例…
            </p>
          )}
          {exampleError && (
            <Alert
              className="notice"
              type="error"
              showIcon
              title={exampleError}
            />
          )}
          {parsedExamples && (
            <div className="parsed-examples">
              <Alert
                type="success"
                showIcon
                title={`已解析 ${parsedExamples.count} 条 JSON 样例`}
                description="instruction 保持参考指令，扩写 input，并生成对应的 output。"
              />
              <details>
                <summary>查看解析后的完整样例</summary>
                <pre>{JSON.stringify(parsedExamples.examples, null, 2)}</pre>
              </details>
            </div>
          )}
          <Form.Item
            name="output_labels"
            label="答案标签约束（可选）"
            extra="从指令中的明确标签列表识别，请核对。多标签答案用顿号分隔；非分类任务可清空。"
          >
            <Select
              mode="tags"
              tokenSeparators={["、", ",", "，"]}
              placeholder="例如：数值查询、变化分析、未来预测"
            />
          </Form.Item>
          <Form.Item
            name="target"
            label="计划新增数量"
            rules={[{ required: true }]}
          >
            <InputNumber
              min={1}
              max={100000}
              precision={0}
              style={{ width: "100%" }}
            />
          </Form.Item>
        </Form>
      </Drawer>
      <Drawer
        title={sampleEdit === "new" ? "人工添加样本" : "编辑样本"}
        size={520}
        open={sampleEdit !== null}
        onClose={() => setSampleEdit(null)}
        footer={
          <div className="drawer-footer">
            <Button onClick={() => setSampleEdit(null)}>取消</Button>
            <Button
              type="primary"
              loading={busy}
              onClick={() => sampleForm.submit()}
            >
              保存为待审核
            </Button>
          </div>
        }
      >
        <Alert
          className="notice"
          type="info"
          showIcon
          title="保存后需要重新审核"
          description="原始指令、问句、答案与修改记录都会保留，已冻结的数据版本不受影响。"
        />
        <Form
          name="sample-editor"
          form={sampleForm}
          layout="vertical"
          onFinish={(values) =>
            perform(async () => {
              if (sampleEdit === "new")
                await api(`/versions/${versionId}/samples`, values);
              else if (sampleEdit)
                await api(`/versions/${versionId}/review`, {
                  ...values,
                  action: "edit",
                  selected: [
                    { id: sampleEdit.id, revision: sampleEdit.revision },
                  ],
                });
              setSampleEdit(null);
            }, "样本已保存，等待审核")
          }
        >
          <Form.Item name="label" label="所属场景" rules={[{ required: true }]}>
            <Select options={labelOptions} />
          </Form.Item>
          {sampleScenario?.format === "alpaca" && (
            <Form.Item
              name="instruction"
              label="任务指令 / instruction"
              rules={[
                { required: true, whitespace: true, message: "请输入任务指令" },
              ]}
            >
              <Input.TextArea rows={4} maxLength={12000} />
            </Form.Item>
          )}
          <Form.Item
            name="text"
            label="问句 / input"
            rules={[
              { required: true, whitespace: true, message: "请输入样本文本" },
            ]}
          >
            <Input.TextArea rows={5} maxLength={4000} showCount />
          </Form.Item>
          {sampleScenario?.format === "alpaca" &&
            (sampleScenario.output_labels?.length ? (
              <Form.Item
                name="output"
                label="答案标签 / output"
                rules={[{ required: true, message: "请选择至少一个答案标签" }]}
                getValueProps={(value: string) => ({
                  value: value ? value.split("、") : [],
                })}
                normalize={(value: string[]) => value.join("、")}
              >
                <Select
                  mode="multiple"
                  options={sampleScenario.output_labels.map((x) => ({
                    value: x,
                    label: x,
                  }))}
                />
              </Form.Item>
            ) : (
              <Form.Item
                name="output"
                label="答案 / output"
                rules={[
                  {
                    required: true,
                    whitespace: true,
                    message: "请输入正确答案",
                  },
                ]}
              >
                <Input.TextArea rows={3} maxLength={4000} />
              </Form.Item>
            ))}
        </Form>
        {sampleEdit && sampleEdit !== "new" && (
          <div className="original-sample">
            <label>原始文本 · {sources[sampleEdit.source]}</label>
            <pre>
              {sampleEdit.original_instruction
                ? JSON.stringify(
                    {
                      instruction: sampleEdit.original_instruction,
                      input: sampleEdit.original,
                      output: sampleEdit.original_output,
                    },
                    null,
                    2,
                  )
                : sampleEdit.original}
            </pre>
          </div>
        )}
      </Drawer>
      <Modal
        title="按场景补生成"
        open={extraOpen}
        onCancel={() => setExtraOpen(false)}
        onOk={() => extraForm.submit()}
        confirmLoading={busy}
        okText="开始补生成"
      >
        <Form
          name="supplement"
          form={extraForm}
          layout="vertical"
          onFinish={(values) =>
            perform(async () => {
              await api(`/versions/${versionId}/jobs`, values);
              setExtraOpen(false);
            }, "补生成任务已启动")
          }
        >
          <Form.Item name="label" label="生成场景" rules={[{ required: true }]}>
            <Select options={labelOptions} />
          </Form.Item>
          <Form.Item
            name="amount"
            label="新增数量"
            rules={[{ required: true }]}
          >
            <InputNumber min={1} max={100000} precision={0} />
          </Form.Item>
        </Form>
      </Modal>
      <Drawer
        title="生成服务配置"
        size={480}
        open={settingsOpen}
        onClose={() => setSettingsOpen(false)}
      >
        <div className="settings-service">
          <CloudServerOutlined />
          <h2>{settings.data?.model}</h2>
          <Tag color="blue">
            {settings.data?.provider === "local"
              ? "本地模型"
              : "兼容 Chat Completions API"}
          </Tag>
        </div>
        <dl className="settings-list">
          <dt>服务地址</dt>
          <dd>{settings.data?.base_url}</dd>
          <dt>配置文件</dt>
          <dd>
            <code>{settings.data?.config_path}</code>
          </dd>
          <dt>密钥状态</dt>
          <dd>
            {settings.data?.api_key_configured
              ? "已配置（尚未验证接口连通性）"
              : "未配置"}
          </dd>
          <dt>密钥环境变量</dt>
          <dd>
            <code>{settings.data?.api_key_env}</code>
          </dd>
        </dl>
        <Alert
          type="info"
          showIcon
          title="换机器时直接替换配置"
          description="在配置文件中设置服务地址和模型，密钥放在环境变量或 .env。新建任务时读取新配置，进行中的任务保留原配置。"
        />
      </Drawer>
    </div>
  );
}

createRoot(document.getElementById("root")!).render(
  <ConfigProvider
    locale={zhCN}
    theme={{
      token: {
        colorPrimary: "#315efb",
        colorText: "#25324b",
        colorTextSecondary: "#7e899c",
        colorBorder: "#e3e8f0",
        colorBgLayout: "#f5f7fb",
        borderRadius: 8,
        fontFamily:
          'Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", "Microsoft YaHei", sans-serif',
        controlHeight: 36,
      },
      components: {
        Table: {
          headerBg: "#f8f9fc",
          headerColor: "#738098",
          cellPaddingBlock: 17,
        },
        Menu: {
          itemSelectedBg: "#eef2ff",
          itemSelectedColor: "#315efb",
          itemHeight: 46,
        },
        Card: { paddingLG: 24 },
        Button: { primaryShadow: "0 3px 8px rgba(49,94,251,.14)" },
      },
    }}
  >
    <App>
      <Workbench />
    </App>
  </ConfigProvider>,
);
