import { Alert, Button, Drawer, Table, Tag } from "antd";
export type LabelTask = {
  id: string;
  status: string;
  error: string;
  attempts: number;
  created: string;
  name?: string;
  project_name?: string;
  input: { examples: Record<string, unknown>[]; description: string };
  snapshot: {
    project_id?: string;
    version_id?: string;
    project_name?: string;
    intents?: unknown[];
    index?: number;
    form: Record<string, unknown>;
  };
  result: {
    classification: boolean;
    labels: string[];
    explanation: string;
  } | null;
};
export type LabelTaskList = {
  rows: LabelTask[];
  total: number;
  active: number;
  page: number;
};
export const labelTaskStatus: Record<string, string> = {
  queued: "排队中",
  running: "推断中",
  completed: "已完成",
  failed: "执行失败",
  interrupted: "已中断",
};
export function stableJSON(value: unknown): string {
  const normalize = (item: unknown): unknown =>
    Array.isArray(item)
      ? item.map(normalize)
      : item !== null && typeof item === "object"
        ? Object.fromEntries(
            Object.entries(item)
              .sort(([a], [b]) => a.localeCompare(b))
              .map(([key, v]) => [key, normalize(v)]),
          )
        : item;
  return JSON.stringify(normalize(value)) || "";
}
export default function LabelTaskDrawer({
  open,
  onClose,
  data,
  error,
  loading,
  page,
  onPage,
  onOpen,
  onRetry,
  onRefresh,
}: {
  open: boolean;
  onClose: () => void;
  data?: LabelTaskList;
  error: string;
  loading: boolean;
  page: number;
  onPage: (page: number) => void;
  onOpen: (id: string) => void;
  onRetry: (id: string) => void;
  onRefresh: () => void;
}) {
  return (
    <Drawer
      title="标签推断任务"
      zIndex={1100}
      size={900}
      open={open}
      onClose={onClose}
    >
      <Alert
        className="notice"
        showIcon
        type="info"
        title="推断任务与草稿持久保存"
        description="任务及提交时的场景草稿保存在服务端工作区。关闭页面不会停止推断；服务重启后未完成的任务标记为中断，可手动重试。恢复草稿后，请核对结果并保存项目配置。"
      />
      <Button onClick={onRefresh}>刷新任务</Button>
      {error && <Alert className="notice" type="error" title={error} />}
      <Table
        rowKey="id"
        loading={loading}
        dataSource={data?.rows || []}
        pagination={{
          current: page,
          total: data?.total || 0,
          pageSize: 20,
          onChange: onPage,
          showSizeChanger: false,
        }}
        columns={[
          {
            title: "场景 / 项目",
            render: (_, row) => (
              <div>
                {row.name}
                <small style={{ display: "block" }}>{row.project_name}</small>
              </div>
            ),
          },
          {
            title: "状态",
            dataIndex: "status",
            render: (status: string) => (
              <Tag>{labelTaskStatus[status] || status}</Tag>
            ),
          },
          {
            title: "推断结果",
            render: (_, row) =>
              row.error || row.result?.explanation || "等待推断结果",
          },
          {
            title: "创建时间",
            dataIndex: "created",
            render: (value: string) => new Date(value).toLocaleString("zh-CN"),
          },
          {
            title: "操作",
            render: (_, row) => (
              <>
                <Button type="link" onClick={() => onOpen(row.id)}>
                  打开草稿与结果
                </Button>
                {["failed", "interrupted"].includes(row.status) && (
                  <Button onClick={() => onRetry(row.id)}>重试推断</Button>
                )}
              </>
            ),
          },
        ]}
      />
    </Drawer>
  );
}
