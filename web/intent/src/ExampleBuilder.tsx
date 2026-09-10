import { useState } from "react";
import {
  App,
  Button,
  Input,
  InputNumber,
  Popconfirm,
  Select,
  Switch,
  Table,
  Tag,
} from "antd";
import { DeleteOutlined, PlusOutlined } from "@ant-design/icons";

type Value =
  string | number | boolean | null | Value[] | { [key: string]: Value };
type RecordValue = { [key: string]: Value };
const templates: { value: string; label: string; record: RecordValue }[] = [
  {
    value: "alpaca",
    label: "任务指令 + 输入 + 答案",
    record: { instruction: "", input: "", output: "" },
  },
  {
    value: "system",
    label: "系统提示 + 任务指令 + 输入 + 答案",
    record: { system: "", instruction: "", input: "", output: "" },
  },
  { value: "qa", label: "问题 + 答案", record: { question: "", answer: "" } },
  {
    value: "chat",
    label: "多轮对话（messages）",
    record: {
      messages: [
        { role: "user", content: "" },
        { role: "assistant", content: "" },
      ],
    },
  },
];
export const defaultExamples = () => [structuredClone(templates[0].record)];
const names: Record<string, string> = {
  system: "系统提示",
  instruction: "任务指令",
  input: "输入",
  output: "答案",
  question: "问题",
  answer: "答案",
  messages: "对话",
  role: "角色",
  content: "内容",
  from: "角色",
  value: "内容",
};
const typeName = (v: Value) =>
  v === null
    ? "null"
    : Array.isArray(v)
      ? "数组"
      : (
          {
            string: "文本",
            number: "数值",
            boolean: "布尔",
            object: "对象",
          } as Record<string, string>
        )[typeof v] || "";
const clearValues = (v: Value): Value => {
  if (typeof v === "string") return "";
  if (Array.isArray(v)) return v.map(clearValues);
  if (v !== null && typeof v === "object")
    return Object.fromEntries(
      Object.entries(v).map(([k, x]) => [
        k,
        ["system", "instruction", "system_prompt", "role", "from"].includes(k)
          ? structuredClone(x)
          : clearValues(x),
      ]),
    );
  return v;
};
function ValueInput({
  value,
  path,
  onChange,
}: {
  value: Value;
  path: string;
  onChange: (v: Value) => void;
}) {
  const key = path.split(".").at(-1)!;
  if (value === null) return <Tag>null（空值）</Tag>;
  if (typeof value === "boolean")
    return <Switch aria-label={path} checked={value} onChange={onChange} />;
  if (typeof value === "number")
    return (
      <InputNumber
        aria-label={path}
        value={value}
        style={{ width: "100%" }}
        onChange={(v) => {
          if (v !== null) onChange(v);
        }}
      />
    );
  if (typeof value === "string") {
    if (["role", "from"].includes(key)) {
      const options =
        key === "role"
          ? ["system", "user", "assistant"]
          : ["system", "human", "gpt"];
      if (!options.includes(value)) options.push(value);
      return (
        <Select
          aria-label={path}
          style={{ width: "100%" }}
          value={value}
          options={options.map((v) => ({
            value: v,
            label:
              {
                system: "系统",
                user: "用户",
                assistant: "助手",
                human: "用户",
                gpt: "助手",
              }[v] || v,
          }))}
          onChange={onChange}
        />
      );
    }
    return (
      <Input.TextArea
        aria-label={path}
        autoSize={{ minRows: 2, maxRows: 8 }}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={`填写${names[key] || key}`}
      />
    );
  }
  if (Array.isArray(value)) {
    const chat = ["messages", "conversations"].includes(key);
    return (
      <div className="builder-nested">
        {value.map((item, i) => (
          <div className="builder-array-item" key={i}>
            <div className="builder-item-heading">
              <span>第 {i + 1} 项</span>
              <Button
                size="small"
                type="text"
                aria-label={`${path} 删除第 ${i + 1} 项`}
                icon={<DeleteOutlined />}
                onClick={() => onChange(value.filter((_, j) => j !== i))}
              />
            </div>
            <ValueInput
              value={item}
              path={`${path}.${i + 1}`}
              onChange={(v) => onChange(value.map((x, j) => (i === j ? v : x)))}
            />
          </div>
        ))}
        {chat ? (
          <Button
            size="small"
            onClick={() =>
              onChange([
                ...value,
                ...(key === "messages"
                  ? [
                      { role: "user", content: "" },
                      { role: "assistant", content: "" },
                    ]
                  : [
                      { from: "human", value: "" },
                      { from: "gpt", value: "" },
                    ]),
              ])
            }
          >
            添加一轮对话
          </Button>
        ) : value.length > 0 ? (
          <Button
            size="small"
            onClick={() => onChange([...value, clearValues(value[0])])}
          >
            添加数组项
          </Button>
        ) : (
          <Tag>空数组 []</Tag>
        )}
      </div>
    );
  }
  return (
    <div className="builder-nested">
      {Object.entries(value).map(([field, v]) => (
        <div key={field} className="builder-object-field">
          <span>
            {names[field] || field} <small>{field}</small>
          </span>
          <ValueInput
            value={v}
            path={`${path}.${field}`}
            onChange={(next) =>
              onChange(
                Object.fromEntries(
                  Object.entries(value).map(([k, x]) => [
                    k,
                    k === field ? next : x,
                  ]),
                ),
              )
            }
          />
        </div>
      ))}
      {!Object.keys(value).length && <Tag>空对象 {"{}"}</Tag>}
    </div>
  );
}
export default function ExampleBuilder({
  initialExamples,
  onChange,
}: {
  initialExamples: RecordValue[];
  onChange: (records: RecordValue[]) => void;
}) {
  const { modal, message } = App.useApp();
  const [records, setRecords] = useState(initialExamples);
  const [index, setIndex] = useState(0);
  const [field, setField] = useState("");
  const [kind, setKind] = useState("string");
  function update(next: RecordValue[]) {
    setRecords(next);
    onChange(next);
  }
  const record = records[index];
  function addField() {
    const key = field.trim();
    if (!key || key.length > 128)
      return void message.error("字段名需要 1 到 128 个字符");
    if (Object.hasOwn(record, key)) return void message.error("字段名已存在");
    const value: Value =
      kind === "number"
        ? 0
        : kind === "boolean"
          ? false
          : kind === "null"
            ? null
            : kind === "array"
              ? [""]
              : "";
    update(records.map((r) => ({ ...r, [key]: structuredClone(value) })));
    setField("");
  }
  return (
    <section className="example-builder" aria-label="表格填写样例">
      <div className="builder-template">
        <span>常用字段模板</span>
        <Select
          aria-label="常用字段模板"
          placeholder="选择模板（会替换当前样例）"
          value={null}
          options={templates.map(({ value, label }) => ({ value, label }))}
          onChange={(v) =>
            modal.confirm({
              title: "使用此模板替换样例？",
              content: "当前样例内容将被替换。仅添加字段请使用下方“添加字段”。",
              okText: "使用模板",
              onOk: () => {
                setIndex(0);
                update([
                  structuredClone(templates.find((t) => t.value === v)!.record),
                ]);
              },
            })
          }
        />
      </div>
      <p className="builder-help">
        直接填写内容，无需编写
        JSON。字段名决定导出结构，增删字段会同步应用到所有样例。
      </p>
      <div className="builder-toolbar">
        <Select
          aria-label="当前样例"
          value={index}
          options={records.map((_, i) => ({
            value: i,
            label: `样例 ${i + 1}`,
          }))}
          onChange={setIndex}
        />
        <Button
          icon={<PlusOutlined />}
          disabled={records.length >= 100}
          onClick={() => {
            update([...records, clearValues(record) as RecordValue]);
            setIndex(records.length);
          }}
        >
          新增样例
        </Button>
        <Popconfirm
          title="删除当前参考样例？"
          onConfirm={() => {
            update(records.filter((_, i) => i !== index));
            setIndex(Math.max(0, index - 1));
          }}
        >
          <Button disabled={records.length === 1} danger>
            删除样例
          </Button>
        </Popconfirm>
        <span>共 {records.length} 条</span>
      </div>
      <Table
        pagination={false}
        size="small"
        rowKey="key"
        dataSource={Object.entries(record).map(([key, value]) => ({
          key,
          value,
        }))}
        columns={[
          {
            title: "字段",
            dataIndex: "key",
            width: 130,
            render: (key: string) => (
              <div className="builder-field-name">
                {names[key] && <span>{names[key]}</span>}
                <code>{key}</code>
              </div>
            ),
          },
          { title: "类型", width: 60, render: (_, row) => typeName(row.value) },
          {
            title: "样例内容",
            render: (_, row) => (
              <ValueInput
                value={row.value}
                path={row.key}
                onChange={(v) =>
                  update(
                    records.map((r, i) =>
                      i === index ? { ...r, [row.key]: v } : r,
                    ),
                  )
                }
              />
            ),
          },
          {
            title: "",
            width: 38,
            render: (_, row) => (
              <Popconfirm
                title={`从所有样例删除字段 ${row.key}？`}
                onConfirm={() =>
                  update(
                    records.map((r) =>
                      Object.fromEntries(
                        Object.entries(r).filter(([key]) => key !== row.key),
                      ),
                    ),
                  )
                }
              >
                <Button
                  size="small"
                  type="text"
                  danger
                  disabled={Object.keys(record).length === 1}
                  aria-label={`删除字段 ${row.key}`}
                  icon={<DeleteOutlined />}
                />
              </Popconfirm>
            ),
          },
        ]}
      />
      <div className="builder-add-field">
        <Input
          aria-label="自定义字段名"
          placeholder="自定义字段名，例如 context"
          value={field}
          onChange={(e) => setField(e.target.value)}
          onPressEnter={(e) => {
            e.preventDefault();
            addField();
          }}
        />
        <Select
          aria-label="新增字段类型"
          value={kind}
          onChange={setKind}
          options={[
            { value: "string", label: "文本" },
            { value: "number", label: "数值" },
            { value: "boolean", label: "布尔" },
            { value: "null", label: "空值 null" },
            { value: "array", label: "文本数组" },
          ]}
        />
        <Button icon={<PlusOutlined />} onClick={addField}>
          添加字段
        </Button>
      </div>
    </section>
  );
}
