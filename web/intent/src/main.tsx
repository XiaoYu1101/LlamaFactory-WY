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

type Intent = {
  label: string;
  name: string;
  description: string;
  examples: string[];
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
  { key: "config", icon: <ApartmentOutlined />, label: "意图配置" },
  { key: "generate", icon: <ExperimentOutlined />, label: "样本生成" },
  { key: "review", icon: <FileDoneOutlined />, label: "数据审核" },
  { key: "versions", icon: <DatabaseOutlined />, label: "数据版本" },
];
const copy: Record<Page, [string, string]> = {
  config: ["定义你的意图分类", "从业务场景出发，配置类别、说明和代表性样例。"],
  generate: [
    "让样例变成数据集",
    "根据意图配置和参考样例生成分类数据，生成进度自动保存。",
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
  const [page, setPage] = useState<Page>("review");
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
    intentForm.setFieldsValue(
      item
        ? { ...item, examples: item.examples.join("\n") }
        : { label: "", name: "", description: "", examples: "", target: 100 },
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
    sampleForm.setFieldsValue(
      sample === "new"
        ? { label: label || intents[0]?.label, text: "" }
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
      title: "样本文本",
      dataIndex: "text",
      render: (text, row) => (
        <div className="sample-text">
          {text}
          {row.conflict && <Tag color="error">类别冲突</Tag>}
        </div>
      ),
    },
    {
      title: "意图类别",
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
              <span className="online-dot" />
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
                      placeholder="例如：售后客服意图分类"
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
                      <strong>{draft.length}</strong> 个类别
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
                    意图类别 <span>{draft.length}</span>
                  </h2>
                  <p>每类至少提供 1 条样例，建议使用真实业务中的不同表达。</p>
                </div>
                <Button
                  icon={<PlusOutlined />}
                  disabled={history || busy}
                  onClick={() => openIntent(draft.length)}
                >
                  添加类别
                </Button>
              </div>
              {!draft.length ? (
                <Card>
                  <Empty description="从第一个意图类别开始">
                    <Button
                      type="primary"
                      icon={<PlusOutlined />}
                      onClick={() => openIntent(0)}
                    >
                      添加意图类别
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
                            title="从配置中移除此类别？"
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
                      <div className="examples-label">
                        参考样例 <span>{intent.examples.length}</span>
                      </div>
                      <div className="seed-preview">
                        {intent.examples.slice(0, 2).map((example, i) => (
                          <p key={i}>
                            <span>0{i + 1}</span>
                            {example}
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
                    均分到各类别
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
              <Empty description="先创建项目并保存意图配置">
                <Button type="primary" onClick={() => setPage("config")}>
                  配置意图类别
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
                        "包含用户样例与生成数据",
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
                        "类别冲突",
                        version.data.counts.conflicts,
                        "相同文本对应多个类别",
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
                        <p>文本修改后会回到待审核状态；勾选样本可批量处理。</p>
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
                        aria-label="搜索样本文本"
                        placeholder="搜索样本文本…"
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
                        aria-label="筛选意图类别"
                        value={label}
                        options={[
                          { value: "", label: "全部意图类别" },
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
                  <Card className="generation-hero">
                    <div>
                      <Tag color="blue">按意图配置生成</Tag>
                      <h2>少量样例，更多业务表达。</h2>
                      <p>
                        根据已保存的类别定义和参考样例生成新样本，完成后即可进入人工审核。
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
                          按类别补生成
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
                    <span>{intents.length} 个意图类别</span>
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
                                {Object.keys(job.targets).length} 个类别
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
                        仅纳入已通过样本。保存后，即使继续修改样本，已有版本也保持不变。
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
                            下载数据集
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
            ? "编辑意图类别"
            : "添加意图类别"
        }
        size={520}
        open={intentEdit !== null}
        onClose={() => setIntentEdit(null)}
        footer={
          <div className="drawer-footer">
            <Button onClick={() => setIntentEdit(null)}>取消</Button>
            <Button type="primary" onClick={() => intentForm.submit()}>
              保存类别
            </Button>
          </div>
        }
      >
        <Form
          name="intent-config"
          form={intentForm}
          layout="vertical"
          onFinish={(values) => {
            const item = {
              ...values,
              examples: values.examples
                .split("\n")
                .map((x: string) => x.trim())
                .filter(Boolean),
            };
            const next = [...draft];
            next[intentEdit!] = item;
            setDraft(next);
            setDirty(true);
            setIntentEdit(null);
          }}
        >
          <Form.Item
            name="name"
            label="类别名称"
            rules={[
              { required: true, whitespace: true, message: "请输入类别名称" },
            ]}
          >
            <Input placeholder="例如：查询退款" maxLength={100} />
          </Form.Item>
          <Form.Item
            name="label"
            label="类别标识"
            extra="训练使用的输出标签，保存后保持一致。"
            rules={[
              {
                required: true,
                pattern: /^[a-zA-Z][a-zA-Z0-9_-]{0,63}$/,
                message:
                  "以字母开头，使用字母、数字、下划线或短横线，最长 64 位",
              },
            ]}
          >
            <Input placeholder="例如：refund_status" />
          </Form.Item>
          <Form.Item
            name="description"
            label="类别说明"
            rules={[
              {
                required: true,
                whitespace: true,
                message: "请描述该类别的适用场景",
              },
            ]}
          >
            <Input.TextArea
              rows={3}
              maxLength={1000}
              placeholder="说明哪些表达属于这个类别，以及与其他类别的边界。"
            />
          </Form.Item>
          <Form.Item
            name="examples"
            label="参考样例"
            extra="每行一条，至少 1 条。建议提供不同表达方式的真实业务样例。"
            rules={[
              {
                required: true,
                whitespace: true,
                message: "请至少提供一条样例",
              },
            ]}
          >
            <Input.TextArea
              rows={6}
              placeholder="我的退款到哪里了？&#10;退款申请通过了，什么时候到账？"
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
          description="原始文本与修改记录会保留，已冻结的数据版本不受影响。"
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
          <Form.Item name="label" label="所属意图" rules={[{ required: true }]}>
            <Select options={labelOptions} />
          </Form.Item>
          <Form.Item
            name="text"
            label="样本文本"
            rules={[
              { required: true, whitespace: true, message: "请输入样本文本" },
            ]}
          >
            <Input.TextArea rows={7} maxLength={1000} showCount />
          </Form.Item>
        </Form>
        {sampleEdit && sampleEdit !== "new" && (
          <div className="original-sample">
            <label>原始文本 · {sources[sampleEdit.source]}</label>
            <p>{sampleEdit.original}</p>
          </div>
        )}
      </Drawer>
      <Modal
        title="按类别补生成"
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
          <Form.Item name="label" label="意图类别" rules={[{ required: true }]}>
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
