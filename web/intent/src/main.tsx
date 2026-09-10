import React, { useEffect, useRef, useState } from "react";
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
import TrainingPanel from "./TrainingPanel";
import { useWorkspaceRoute } from "./useWorkspaceRoute";
import LabelTaskDrawer, {
  labelTaskStatus,
  stableJSON,
} from "./LabelTaskDrawer";
import type { LabelTask, LabelTaskList } from "./LabelTaskDrawer";
import ExampleBuilder, { defaultExamples } from "./ExampleBuilder";

type JSONValue =
  string | number | boolean | null | JSONValue[] | { [key: string]: JSONValue };
type TrainingRecord = { [key: string]: JSONValue };
type RecordSchema = { type: string; properties?: Record<string, RecordSchema> };
type TrainingMapping = Partial<
  Record<"prompt" | "query" | "response" | "system", string>
>;
const jsonScenario = (item?: Intent) =>
  item?.format === "alpaca" || item?.format === "json";
const displayValue = (value: JSONValue | undefined) =>
  typeof value === "string" ? value : JSON.stringify(value, null, 2);
const recordFields = (record: TrainingRecord) =>
  Object.fromEntries(
    Object.entries(record).map(([key, value]) => [
      key,
      value !== null && typeof value === "object"
        ? JSON.stringify(value, null, 2)
        : value,
    ]),
  );
const fieldLabel = (key: string) =>
  ({
    system: "系统提示 / system",
    instruction: "任务指令 / instruction",
    input: "输入 / input",
    output: "答案 / output",
    messages: "对话 / messages",
  })[key] || key;
type ParsedExamples = {
  examples: TrainingRecord[];
  count: number;
  output_labels: string[];
  schema: RecordSchema;
  training_mapping: TrainingMapping | null;
};
type Intent = {
  label: string;
  name: string;
  description: string;
  examples: (TrainingRecord | string)[];
  format?: "alpaca" | "json" | "text";
  schema?: RecordSchema;
  training_mapping?: TrainingMapping | null;
  output_labels?: string[];
  label_task_id?: string;
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
  record?: TrainingRecord | null;
  original_record?: TrainingRecord | null;
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
  manifest: {
    count: number;
    training_ready?: boolean;
    training_note?: string;
    training_datasets?: string[];
  };
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
  api_key_required?: boolean;
  config_path: string;
};
type Page = "config" | "generate" | "review" | "versions" | "train" | "eval";
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
  { key: "train", icon: <ExperimentOutlined />, label: "模型训练" },
  { key: "eval", icon: <CheckCircleOutlined />, label: "模型评测" },
];
const copy: Record<Page, [string, string]> = {
  train: [
    "模型微调配置",
    "配置模型微调任务，并通过 LlamaFactory 查看训练状态与结果。",
  ],
  eval: ["评测与预测", "配置独立评测数据，使用 LlamaFactory 执行评测或预测。"],
  config: [
    "生成场景配置",
    "通过表格或 JSON 文件配置参考样例，按样例字段、值类型及嵌套结构生成数据。",
  ],
  generate: [
    "样本生成任务",
    "根据场景要求与参考样例生成数据；结构校验通过的新增样本进入待审核列表。",
  ],
  review: [
    "样本审核与维护",
    "查看并修订生成内容，审核样本的任务相关性、标注准确性与数据完整性。",
  ],
  versions: [
    "数据集版本与导出",
    "将审核通过的样本保存为固定数据集版本，导出数据及 LlamaFactory 字段映射配置。",
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
  const { page, setPage, projectId, setProjectId, versionId, setVersionId } =
    useWorkspaceRoute();
  const [tick, setTick] = useState(0);
  const [poll, setPoll] = useState(0);
  const [busy, setBusy] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [discoveredModels, setDiscoveredModels] = useState<string[]>([]);
  const [initial, setInitial] = useState(true);
  const [draftName, setDraftName] = useState("");
  const [draft, setDraft] = useState<Intent[]>([]);
  const [dirty, setDirty] = useState(false);
  const [intentEdit, setIntentEdit] = useState<number | null>(null);
  const [intentForm] = Form.useForm();
  const restoredDraft = useRef(false);
  const [labelTasksOpen, setLabelTasksOpen] = useState(false);
  const [labelTasksPage, setLabelTasksPage] = useState(1);
  const [labelTaskPoll, setLabelTaskPoll] = useState(0);
  const [activeLabelId, setActiveLabelId] = useState("");
  const [appliedLabelId, setAppliedLabelId] = useState("");
  const [labelStarting, setLabelStarting] = useState(false);
  const descriptionText = Form.useWatch("description", intentForm);

  const exampleText = Form.useWatch("examples", intentForm);
  const mappingMode = Form.useWatch("mapping_mode", intentForm);
  const [exampleMode, setExampleMode] = useState<"json" | "table">("json");
  const [tableExamples, setTableExamples] = useState<TrainingRecord[]>([]);
  const [builderVersion, setBuilderVersion] = useState(0);
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
  const labelTasks = useData<LabelTaskList>(
    `/label-tasks?page=${labelTasksPage}`,
    tick + labelTaskPoll,
  );
  const activeLabelTask = useData<LabelTask>(
    activeLabelId ? `/label-tasks/${activeLabelId}` : null,
    tick + labelTaskPoll,
  );
  const labelInFlight =
    labelStarting ||
    (!!activeLabelId &&
      !activeLabelTask.error &&
      (!activeLabelTask.data ||
        ["queued", "running"].includes(activeLabelTask.data.status)));
  const sameLabelInput =
    !!activeLabelTask.data &&
    !!parsedExamples &&
    stableJSON(activeLabelTask.data.input) ===
      stableJSON({
        examples: parsedExamples.examples,
        description: (descriptionText || "").trim(),
      });
  const labelReady =
    activeLabelTask.data?.status === "completed" && sameLabelInput;
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
  const firstExample = sampleScenario?.examples[0];
  const sampleTemplate: TrainingRecord =
    sampleEdit && sampleEdit !== "new" && sampleEdit.label === sampleLabel
      ? sampleEdit.record || {
          instruction: sampleEdit.instruction,
          input: sampleEdit.text,
          output: sampleEdit.output,
        }
      : typeof firstExample === "object"
        ? firstExample
        : {};
  const answerField = ["output", "answer", "response", "completion"].find(
    (key) => typeof sampleTemplate[key] === "string",
  );
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
      if (!projectId && projects.data.length) setProjectId(projects.data[0].id);
      setInitial(false);
    }
  }, [projects.data, initial]);
  useEffect(() => {
    if (project.data && !restoredDraft.current) {
      if (!versionId) setVersionId(project.data.current_version);
      setDraftName(project.data.name);
    }
  }, [project.data?.id, project.data?.current_version, versionId]);
  useEffect(() => {
    if (version.data && !restoredDraft.current) {
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

          if (intentForm.getFieldValue("mapping_mode") !== "custom")
            intentForm.setFieldValue(
              "training_mapping",
              result.training_mapping || {},
            );
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

  useEffect(() => {
    if (!labelTasksOpen && !labelTasks.data?.active && !labelInFlight) return;
    const timer = window.setInterval(
      () => setLabelTaskPoll((v) => v + 1),
      2000,
    );
    return () => window.clearInterval(timer);
  }, [labelTasksOpen, labelTasks.data?.active, labelInFlight]);
  useEffect(() => {
    if (
      intentEdit !== null &&
      labelReady &&
      activeLabelTask.data?.result &&
      appliedLabelId !== activeLabelId
    ) {
      intentForm.setFieldsValue({
        output_labels: activeLabelTask.data.result.labels,
        label_task_id: activeLabelId,
      });
      setAppliedLabelId(activeLabelId);
    }
  }, [
    intentEdit,
    labelReady,
    activeLabelTask.data,
    activeLabelId,
    appliedLabelId,
  ]);
  async function startLabelInference() {
    if (!parsedExamples || parsedExamples.count < 4 || labelInFlight) return;
    try {
      await intentForm.validateFields(["name", "label", "description"]);
      setLabelStarting(true);
      const form = intentForm.getFieldsValue(true);
      const task = await api<LabelTask>("/label-tasks", {
        examples: parsedExamples.examples,
        description: form.description,
        snapshot: {
          project_id: projectId,
          version_id: versionId,
          project_name: draftName,
          intents: draft,
          index: intentEdit,
          form,
        },
      });
      setActiveLabelId(task.id);
      setAppliedLabelId("");
      setLabelTaskPoll((v) => v + 1);
      message.success(
        "推断任务已保存，可以关闭页面，稍后从标签推断任务入口恢复",
      );
    } catch (error) {
      if (error instanceof Error) message.error(error.message);
    } finally {
      setLabelStarting(false);
    }
  }
  async function retryLabelInference(id: string) {
    await perform(async () => {
      await api(`/label-tasks/${id}/retry`, {});
      setLabelTaskPoll((v) => v + 1);
    }, "已重新提交推断任务");
  }
  function restoreLabelTask(id: string) {
    switchContext(() => {
      void perform(async () => {
        const task = await api<LabelTask>(`/label-tasks/${id}`);
        const saved = task.snapshot;
        restoredDraft.current = true;
        setInitial(false);
        setProjectId(saved.project_id || "");
        setVersionId(saved.version_id || "");
        setDraftName(saved.project_name || "");
        setDraft((saved.intents || []) as Intent[]);
        setPage("config");
        setDirty(true);
        setExampleMode("json");
        setIntentEdit(saved.index ?? 0);
        intentForm.resetFields();
        intentForm.setFieldsValue({
          ...saved.form,
          examples: JSON.stringify(task.input.examples, null, 2),
          description: task.input.description,
        });
        setActiveLabelId(task.id);
        setAppliedLabelId("");
        setLabelTasksOpen(false);
      }, "任务草稿已恢复，请核对场景内容并保存项目配置");
    });
  }
  function switchContext(action: () => void) {
    if (dirty)
      modal.confirm({
        title: "有未保存的配置",
        zIndex: 1200,
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
      restoredDraft.current = false;
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
    setExampleMode("json");
    const item = draft[index];
    setActiveLabelId(item?.label_task_id || "");
    setAppliedLabelId(item?.label_task_id || "");
    intentForm.resetFields();
    intentForm.setFieldsValue(
      item
        ? {
            ...item,
            mapping_mode: item.training_mapping ? "custom" : "auto",
            examples: item.examples
              .map((x) => (typeof x === "string" ? x : JSON.stringify(x)))
              .join("\n"),
          }
        : {
            label: "",
            name: "",
            description: "",
            examples: "",
            target: 1000,
            mapping_mode: "auto",
          },
    );
  }
  async function removeScenario(index: number) {
    const scenario = draft[index];
    const persisted = version.data?.config.some(
      (item) => item.label === scenario.label,
    );
    if (!persisted) {
      setDraft(draft.filter((_, i) => i !== index));
      setDirty(true);
      return;
    }
    if (dirty) {
      message.error("请先保存其他配置修改，再删除已保存的场景。");
      return;
    }
    await perform(async () => {
      const result = await api<{ version_id: string; config: Intent[] }>(
        `/versions/${versionId}/delete-scenario`,
        { label: scenario.label },
      );
      restoredDraft.current = false;
      setVersionId(result.version_id);
      setDraft(result.config);
      setDirty(false);
      setSelected([]);
    }, "场景及关联样本已删除并保存");
  }
  function clearAllSamples() {
    modal.confirm({
      title: "清空当前版本的全部审核样本？",
      content:
        "永久删除所有分页的待审核、通过、驳回及回收站样本，不受当前筛选影响；场景参考样例保留。暂停的生成任务将取消。已导出的固定数据包不受影响。此操作不可恢复。",
      okText: "确认清空",
      okButtonProps: { danger: true },
      onOk: () =>
        perform(async () => {
          await api(`/versions/${versionId}/clear-samples`, {});
          setSelected([]);
          setTablePage(1);
        }, "当前版本样本已全部清空"),
    });
  }
  function bulkReview(exportJson: boolean) {
    modal.confirm({
      title: exportJson ? "通过全部待审核样本并导出？" : "通过全部待审核样本？",
      content: `将当前版本全部 ${count("pending")} 条待审核样本标记为通过，不受分页和筛选限制。已驳回和已删除样本保持原状态。请确认内容符合要求。`,
      okText: exportJson ? "通过并导出" : "全部通过",
      onOk: () =>
        perform(
          async () => {
            const result = await api<{ id?: string }>(
              `/versions/${versionId}/${exportJson ? "approve-export" : "approve-all"}`,
              {},
            );
            setSelected([]);
            if (exportJson && result.id) {
              const anchor = document.createElement("a");
              anchor.href = `/intent-api/exports/${result.id}/json`;
              anchor.download = "train.json";
              document.body.appendChild(anchor);
              anchor.click();
              anchor.remove();
              setPage("versions");
            }
          },
          exportJson
            ? "全部待审核样本已通过，JSON 已导出"
            : "全部待审核样本已通过",
        ),
    });
  }
  async function changeExampleMode(mode: "json" | "table") {
    if (mode === exampleMode) return;
    if (mode === "table") {
      try {
        const text = intentForm.getFieldValue("examples") || "";
        const rows = text.trim()
          ? (await api<ParsedExamples>("/examples/parse", { examples: text }))
              .examples
          : defaultExamples();
        setTableExamples(rows);
        setBuilderVersion((v) => v + 1);
        intentForm.setFieldValue("examples", JSON.stringify(rows, null, 2));
      } catch (error) {
        message.error(`请先修正 JSON，再切换表格：${(error as Error).message}`);
        return;
      }
    }
    setExampleMode(mode);
  }
  async function saveProject() {
    await perform(async () => {
      const result = await api<{ id: string; version_id: string }>(
        "/projects",
        {
          name: draftName,
          intents: draft,
          project_id: projectId || null,
          base_version_id: versionId || null,
        },
      );
      restoredDraft.current = false;
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
    const record =
      sample === "new"
        ? typeof reference === "object"
          ? reference
          : {}
        : sample.record || {
            instruction: sample.instruction,
            input: sample.text,
            output: sample.output,
          };
    sampleForm.setFieldsValue(
      sample === "new"
        ? { label: scenario?.label, text: "", fields: recordFields(record) }
        : { ...sample, fields: recordFields(record) },
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
      title: "记录摘要",
      dataIndex: "text",
      render: (text, row) => (
        <div className="sample-text">
          <div>{text}</div>
          {row.record && (
            <details className="instruction-detail">
              <summary>查看完整 JSON</summary>
              <pre>{JSON.stringify(row.record, null, 2)}</pre>
            </details>
          )}
          {row.conflict && <Tag color="error">答案冲突</Tag>}
        </div>
      ),
    },
    {
      title: "答案摘要",
      dataIndex: "output",
      width: 220,
      render: (value, row) =>
        value ? (
          <span className="answer-text">{value}</span>
        ) : row.record ? (
          <span className="muted">空字符串</span>
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
              title="将该样本移至回收站？"
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
          <strong>训练数据准备流程</strong>
          <p>
            场景配置、样本生成、
            <br />
            数据审核与版本管理。
          </p>
          <div className="guide-steps">
            <span>配置</span>
            <i />
            <span>生成</span>
            <i />
            <span>审核</span>
          </div>
        </div>
        <Button type="text" onClick={() => setLabelTasksOpen(true)}>
          标签推断任务
          {labelTasks.data?.active
            ? `（${labelTasks.data.active} 进行中）`
            : ""}
        </Button>
        <button
          className="sidebar-settings"
          onClick={() => setSettingsOpen(true)}
        >
          <SettingOutlined /> 生成服务配置
        </button>
        <div className="sidebar-foot">
          <span className="online-dot" /> 服务端工作区 <span>v1.0</span>
        </div>
      </aside>
      <div className="main-shell">
        <header className="topbar">
          <div className="workspace-label">
            <FolderOpenOutlined />
            <span>项目工作区</span>
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
                  restoredDraft.current = false;
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
              {settings.data?.model || "正在加载配置"}
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
                      restoredDraft.current = false;
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
                  <Empty description="尚未配置生成场景">
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
                            title="删除场景及关联样本？"
                            okText="确认删除"
                            description="删除将立即生效，并永久清理本项目各配置版本中该场景的样本；相关暂停任务将取消。已导出的固定数据包保留，此操作不可恢复。"
                            onConfirm={() => removeScenario(index)}
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
                      {!jsonScenario(intent) && (
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
                                {Object.entries(example).map(([key, value]) => (
                                  <React.Fragment key={key}>
                                    <strong>{key}</strong> {displayValue(value)}
                                    <br />
                                  </React.Fragment>
                                ))}
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
                    按场景平均分配
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
          ) : page === "train" || page === "eval" ? (
            <TrainingPanel
              mode={page}
              integrated={!!settings.data?.integrated}
            />
          ) : !versionId ? (
            <Card>
              <Empty description="先创建项目并应用场景配置配置">
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
                        "当前版本中未删除的样本；不包含 JSON 参考样例",
                        <DatabaseOutlined />,
                      ],
                      [
                        "待审核",
                        count("pending"),
                        "尚未完成审核的样本",
                        <FileDoneOutlined />,
                      ],
                      [
                        "已通过",
                        count("approved"),
                        "已通过人工审核的样本",
                        <CheckCircleOutlined />,
                      ],
                      [
                        "答案冲突",
                        version.data.counts.conflicts,
                        "相同任务指令和输入对应不同输出",
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
                          所有字段均可查看和修改，保存后重新审核；数组和嵌套对象使用
                          JSON 编辑。
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
                    <div className="example-mode-switch">
                      <Button
                        disabled={busy || !count("pending")}
                        onClick={() => bulkReview(false)}
                      >
                        通过全部待审核样本
                      </Button>
                      <Button
                        danger
                        disabled={running || busy}
                        onClick={clearAllSamples}
                      >
                        清空当前版本样本
                      </Button>
                      <Button
                        type="primary"
                        icon={<DownloadOutlined />}
                        disabled={
                          busy || !(count("pending") + count("approved"))
                        }
                        onClick={() => bulkReview(true)}
                      >
                        全部通过并导出 JSON
                      </Button>
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
                      <FileDoneOutlined />{" "}
                      处理完全部待审核样本后，创建数据集版本
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
                    !settings.data.api_key_configured &&
                    settings.data.api_key_required !== false && (
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
                      <h2>按参考样例生成数据</h2>
                      <p>
                        根据已保存的 JSON
                        样例生成完整训练记录。仅统计结构校验通过的新增样本；重复或校验失败的结果会被过滤，内容仍需人工审核。
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
                          按场景追加生成
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
                      <p>
                        支持暂停与恢复。关闭浏览器页面不会停止服务端生成任务。
                      </p>
                    </div>
                    <Tag>{version.data.jobs.length} 个任务</Tag>
                  </div>
                  {!version.data.jobs.length ? (
                    <Card>
                      <Empty description="暂无生成任务，请应用场景配置配置后创建任务" />
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
                      <h2>创建固定数据集版本</h2>
                      <p>
                        仅纳入审核通过且未删除的样本，保留参考样例定义的全部字段、值类型及嵌套结构。
                        JSON 参考样例不自动纳入训练集；导出记录统一保存为 JSON
                        数组。
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
                          "数据集版本已创建",
                        )
                      }
                    >
                      创建数据集版本
                    </Button>
                  </Card>
                  <div className="section-heading">
                    <div>
                      <h2>已创建的数据集版本</h2>
                      <p>
                        数据包包含
                        train.json（训练数据）、dataset_info.json（字段映射）、reviewed_samples.json（样本追溯信息）和
                        manifest.json（版本清单）。LoRA
                        参数需在模型训练页单独配置。
                      </p>
                    </div>
                  </div>
                  {!version.data.exports.length ? (
                    <Card>
                      <Empty description="暂无固定数据集版本" />
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
                              {number(item.manifest.count)} 条样本 · JSON
                              记录结构保留
                            </p>
                            <code>{item.id.slice(0, 8)}</code>
                          </div>
                          <Button
                            icon={<DownloadOutlined />}
                            href={`/intent-api/exports/${item.id}/download`}
                          >
                            下载数据包 ZIP
                          </Button>
                          <Button
                            icon={<DownloadOutlined />}
                            href={`/intent-api/exports/${item.id}/json`}
                          >
                            下载训练数据 JSON
                          </Button>
                        </div>
                        {item.manifest.training_ready !== undefined && (
                          <Alert
                            type={
                              item.manifest.training_ready
                                ? "success"
                                : "warning"
                            }
                            showIcon
                            title={
                              item.manifest.training_ready
                                ? "训练字段映射已配置"
                                : "可导出 JSON；原生训练需先配置字段映射"
                            }
                            description={item.manifest.training_note}
                          />
                        )}
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
                        ? "使用 LlamaFactory 执行模型微调"
                        : "在训练服务器配置微调任务"
                    }
                    description={
                      settings.data?.integrated
                        ? "在模型训练页配置并保存 LoRA 参数，再通过完整界面“意图分类”页下方的配置入口填入原生训练页。核对参数后启动训练。"
                        : "在模型训练页保存配置并下载 LoRA 训练包，或在完整界面按 dataset_info.json 登记的数据集名称配置训练。数据包本身不包含 LoRA 参数。"
                    }
                  />
                </>
              )}
            </>
          )}
          <footer className="page-footer">
            LlamaFactory WY <span>训练数据管理工作台</span>
          </footer>
        </main>
      </div>
      <LabelTaskDrawer
        open={labelTasksOpen}
        onClose={() => setLabelTasksOpen(false)}
        data={labelTasks.data}
        error={labelTasks.error}
        loading={labelTasks.loading}
        page={labelTasksPage}
        onPage={setLabelTasksPage}
        onOpen={restoreLabelTask}
        onRetry={retryLabelInference}
        onRefresh={() => setLabelTaskPoll((v) => v + 1)}
      />
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
              disabled={
                parsing ||
                !!exampleError ||
                !parsedExamples ||
                parsedExamples.count < 4 ||
                !labelReady ||
                appliedLabelId !== activeLabelId ||
                labelStarting
              }
              onClick={() => intentForm.submit()}
            >
              应用场景配置
            </Button>
          </div>
        }
      >
        <Form
          name="intent-config"
          disabled={labelInFlight}
          form={intentForm}
          layout="vertical"
          onFinish={async (values) => {
            try {
              if (!labelReady || appliedLabelId !== activeLabelId)
                throw new Error("请先完成当前样例的标签推断");
              const result = await api<ParsedExamples>("/examples/parse", {
                examples: values.examples,
              });
              const item: Intent = {
                ...values,
                format: "json",
                label_task_id: activeLabelId,
                training_mapping:
                  values.mapping_mode === "custom"
                    ? values.training_mapping
                    : null,
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
          <Alert
            className="notice"
            type="info"
            showIcon
            title="场景配置保存范围"
            description="应用场景配置后，修改保留在当前项目草稿中。请在场景配置页点击“保存配置”，将修改保存到服务端。"
          />
          <Form.Item
            name="name"
            label="场景名称"
            rules={[
              { required: true, whitespace: true, message: "请输入场景名称" },
            ]}
          >
            <Input placeholder="例如：客户服务意图分类" maxLength={100} />
          </Form.Item>
          <Form.Item
            name="label"
            label="场景标识"
            extra="用于区分生成场景，例如 customer_intent；记录内容和字段以参考 JSON 为准。"
            rules={[
              {
                required: true,
                pattern: /^[a-zA-Z][a-zA-Z0-9_-]{0,63}$/,
                message:
                  "以字母开头，使用字母、数字、下划线或短横线，最长 64 位",
              },
            ]}
          >
            <Input placeholder="例如：customer_intent" />
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
              placeholder="请说明任务目标、业务范围、内容要求及输出约束；分类任务可补充标签定义与边界。"
            />
          </Form.Item>
          <div
            className="example-mode-switch"
            role="group"
            aria-label="样例输入方式"
          >
            <Button
              type={exampleMode === "table" ? "primary" : "default"}
              onClick={() => void changeExampleMode("table")}
            >
              表格填写
            </Button>
            <Button
              type={exampleMode === "json" ? "primary" : "default"}
              onClick={() => void changeExampleMode("json")}
            >
              JSON / 文件导入
            </Button>
          </div>
          {exampleMode === "table" && (
            <ExampleBuilder
              key={builderVersion}
              initialExamples={tableExamples}
              onChange={(rows) => {
                intentForm.setFieldValue(
                  "examples",
                  JSON.stringify(rows, null, 2),
                );
              }}
            />
          )}
          {exampleMode === "json" && (
            <label className="json-file-import">
              导入 JSON / JSONL 文件
              <input
                aria-label="导入 JSON 样例文件"
                disabled={labelInFlight}
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
          )}
          <Form.Item
            hidden={exampleMode === "table"}
            name="examples"
            label="参考样例"
            extra="支持每行一个 JSON 对象（JSONL）或 JSON 数组。字段名、值类型和嵌套结构由样例决定，同场景样例结构需一致。"
            rules={[
              {
                required: true,
                whitespace: true,
                message: "请提供至少 4 条不同的样例",
              },
            ]}
          >
            <Input.TextArea
              rows={10}
              className="json-editor"
              placeholder='{"instruction":"任务指令","input":"参考问句","output":"正确答案"}'
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
          {parsedExamples && parsedExamples.count < 4 && (
            <Alert
              className="notice"
              showIcon
              type="warning"
              title={`至少需要 4 条不同的参考样例，当前 ${parsedExamples.count} 条`}
              description="最多支持 100 条参考样例。默认每批选取 3 条样例并生成最多 10 条新记录，优先使用历史引用次数较少的样例。"
            />
          )}
          {parsedExamples && (
            <div className="parsed-examples">
              <Alert
                type="success"
                showIcon
                title={`已解析 ${parsedExamples.count} 条 JSON 样例`}
                description="生成时校验全部字段名、值类型与嵌套结构；任务指令和系统上下文保持参考原文，其余内容按业务要求扩写。"
              />
              <details>
                <summary>查看解析后的完整样例</summary>
                <pre>{JSON.stringify(parsedExamples.examples, null, 2)}</pre>
              </details>
            </div>
          )}
          {parsedExamples && (
            <div className="schema-fields">
              <strong>识别到的字段</strong>
              {Object.entries(parsedExamples.schema.properties || {}).map(
                ([key, field]) => (
                  <Tag key={key}>
                    {key}: {field.type}
                  </Tag>
                ),
              )}
            </div>
          )}
          <details className="training-mapping">
            <summary>训练字段映射（自动识别或自定义）</summary>
            <p>
              导出 JSON 保留全部字段。训练字段映射用于指定 LlamaFactory
              读取哪些字段；支持标准对话格式自动识别。
            </p>
            <Form.Item name="mapping_mode" label="映射方式">
              <Select
                options={[
                  { value: "auto", label: "自动识别" },
                  { value: "custom", label: "自定义文本字段" },
                ]}
              />
            </Form.Item>
            {mappingMode === "custom" &&
              (["prompt", "query", "response", "system"] as const).map(
                (key) => (
                  <Form.Item
                    key={key}
                    name={["training_mapping", key]}
                    label={
                      {
                        prompt: "问题 / 指令字段",
                        query: "附加输入字段（可选）",
                        response: "答案字段",
                        system: "系统提示字段（可选）",
                      }[key]
                    }
                  >
                    <Select
                      allowClear
                      options={Object.entries(
                        parsedExamples?.schema.properties || {},
                      )
                        .filter(([, v]) => v.type === "string")
                        .map(([k]) => ({ value: k, label: k }))}
                    />
                  </Form.Item>
                ),
              )}
          </details>
          <div className="label-inference-status">
            <Alert
              className="notice"
              showIcon
              type={
                labelReady
                  ? "success"
                  : activeLabelTask.data?.status === "failed"
                    ? "error"
                    : "info"
              }
              title={
                labelReady
                  ? activeLabelTask.data?.result?.classification
                    ? "标签推断已完成，请核对结果"
                    : "模型判定为非分类任务，可不设置标签约束"
                  : labelInFlight
                    ? `标签${labelTaskStatus[activeLabelTask.data?.status || "queued"]}`
                    : "尚未执行输出标签推断"
              }
              description={
                labelReady
                  ? activeLabelTask.data?.result?.explanation
                  : activeLabelTask.error ||
                    activeLabelTask.data?.error ||
                    (activeLabelTask.data && !sameLabelInput
                      ? "样例或生成要求已变化，旧结果已失效，需要重新推断。"
                      : "推断完成前不能编辑标签或应用场景配置。任务会保存草稿，关闭页面后从左侧“标签推断任务”恢复。")
              }
            />
            <Button
              onClick={startLabelInference}
              loading={labelInFlight}
              disabled={
                labelInFlight ||
                parsing ||
                !parsedExamples ||
                parsedExamples.count < 4
              }
            >
              {activeLabelId ? "重新推断标签" : "开始推断标签"}
            </Button>
            <Button
              disabled={false}
              type="link"
              onClick={() => setLabelTasksOpen(true)}
            >
              查看标签推断任务
            </Button>
          </div>
          <Form.Item
            name="output_labels"
            label="答案标签约束（可选）"
            extra="由大模型推断，完成后可输入新标签并按 Enter 添加，也可删除或清空；修改样例或生成要求后需重新推断。"
          >
            <Select
              mode="tags"
              disabled={!labelReady || labelInFlight}
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
        size={760}
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
          description="保存后样本状态变为待审核。保留原始记录和本次修订记录；已导出的固定数据集版本不受影响。"
        />
        <Form
          name="sample-editor"
          form={sampleForm}
          layout="vertical"
          onFinish={(values) =>
            perform(async () => {
              const payload: {
                label: string;
                text?: string;
                record?: TrainingRecord;
              } = { label: values.label };
              if (jsonScenario(sampleScenario)) {
                const record: TrainingRecord = {};
                for (const [key, template] of Object.entries(sampleTemplate)) {
                  const value = values.fields?.[key];
                  record[key] =
                    template === null
                      ? null
                      : typeof template === "object"
                        ? JSON.parse(value)
                        : value;
                }
                payload.record = record;
              } else payload.text = values.text;
              if (sampleEdit === "new")
                await api(`/versions/${versionId}/samples`, payload);
              else if (sampleEdit)
                await api(`/versions/${versionId}/review`, {
                  ...payload,
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
            <Select
              options={labelOptions}
              onChange={(value) => {
                const example = intents.find((x) => x.label === value)
                  ?.examples[0];
                if (typeof example === "object")
                  sampleForm.setFieldValue("fields", recordFields(example));
              }}
            />
          </Form.Item>
          {jsonScenario(sampleScenario) ? (
            Object.entries(sampleTemplate).map(([key, template]) => {
              if (template === null)
                return (
                  <div className="null-field" key={key}>
                    <label>{key}</label> <Tag>null</Tag>
                  </div>
                );
              const nested = typeof template === "object";
              return (
                <Form.Item
                  key={`${sampleLabel}:${key}`}
                  name={["fields", key]}
                  label={fieldLabel(key)}
                  valuePropName={
                    typeof template === "boolean" ? "checked" : "value"
                  }
                  rules={
                    nested
                      ? [
                          {
                            validator: async (_, value) => {
                              try {
                                JSON.parse(value);
                              } catch {
                                throw new Error("请输入有效 JSON");
                              }
                            },
                          },
                        ]
                      : []
                  }
                  getValueProps={
                    key === answerField && sampleScenario?.output_labels?.length
                      ? (value: string) => ({
                          value: value ? value.split("、") : [],
                        })
                      : undefined
                  }
                  normalize={
                    key === answerField && sampleScenario?.output_labels?.length
                      ? (value: string[]) => value.join("、")
                      : undefined
                  }
                >
                  {typeof template === "boolean" ? (
                    <Switch />
                  ) : typeof template === "number" ? (
                    <InputNumber style={{ width: "100%" }} />
                  ) : key === answerField &&
                    sampleScenario?.output_labels?.length ? (
                    <Select
                      mode="multiple"
                      options={sampleScenario.output_labels.map((x) => ({
                        value: x,
                        label: x,
                      }))}
                    />
                  ) : (
                    <Input.TextArea
                      rows={
                        nested
                          ? 9
                          : ["system", "instruction", "system_prompt"].includes(
                                key,
                              )
                            ? 4
                            : 3
                      }
                      className={nested ? "json-editor" : undefined}
                      maxLength={nested ? 64000 : 16000}
                    />
                  )}
                </Form.Item>
              );
            })
          ) : (
            <Form.Item
              name="text"
              label="样本文本"
              rules={[{ required: true }]}
            >
              <Input.TextArea rows={5} />
            </Form.Item>
          )}
        </Form>
        {sampleEdit && sampleEdit !== "new" && (
          <div className="original-sample">
            <label>原始完整记录 · {sources[sampleEdit.source]}</label>
            <pre>
              {sampleEdit.original_record
                ? JSON.stringify(sampleEdit.original_record, null, 2)
                : sampleEdit.original}
            </pre>
          </div>
        )}
      </Drawer>
      <Modal
        title="按场景追加生成"
        open={extraOpen}
        onCancel={() => setExtraOpen(false)}
        onOk={() => extraForm.submit()}
        confirmLoading={busy}
        okText="开始追加生成"
      >
        <Form
          name="supplement"
          form={extraForm}
          layout="vertical"
          onFinish={(values) =>
            perform(async () => {
              await api(`/versions/${versionId}/jobs`, values);
              setExtraOpen(false);
            }, "追加生成任务已启动")
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
          <dt>配置来源</dt>
          <dd>系统环境变量 → 项目 .env → JSON 配置文件</dd>
          <dt>备用配置文件</dt>
          <dd>
            <code>{settings.data?.config_path}</code>
          </dd>
          <dt>密钥状态</dt>
          <dd>
            {settings.data?.api_key_configured
              ? "已配置（尚未验证接口连通性）"
              : settings.data?.api_key_required === false
                ? "未配置（仅适用于无需鉴权的本机或内网服务）"
                : "未配置"}
          </dd>
          <dt>密钥环境变量</dt>
          <dd>
            <code>{settings.data?.api_key_env}</code>
          </dd>
        </dl>
        <Button
          loading={busy}
          disabled={settings.data?.provider === "local"}
          onClick={() =>
            perform(async () => {
              setDiscoveredModels([]);
              const result = await api<{ models: string[] }>(
                "/settings/discover",
                {},
              );
              setDiscoveredModels(result.models);
            }, "服务模型已检测")
          }
        >
          检测服务模型
        </Button>
        {discoveredModels.length > 0 && (
          <Alert
            className="notice"
            type="success"
            showIcon
            title={
              discoveredModels.length === 1
                ? "检测到一个模型，auto 模式将使用该模型 ID"
                : "检测到多个模型，请在 WY_INTENT_MODEL 中指定用于文本生成的完整模型 ID"
            }
            description={
              <div>
                {discoveredModels.map((name) => (
                  <div key={name}>
                    <code>{name}</code>
                  </div>
                ))}
              </div>
            }
          />
        )}
        <Alert
          type="info"
          showIcon
          title="可在 .env 中配置生成模型"
          description="设置 WY_INTENT_BASE_URL、WY_INTENT_MODEL 和 WY_INTENT_API_KEY。模型名留空或填 auto 时，启动生成前检测服务：只有一个模型则自动使用，多个模型需填写完整名称。新建任务读取新配置。"
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
