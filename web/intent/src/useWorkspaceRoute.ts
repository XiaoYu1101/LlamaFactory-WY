import { useEffect, useState } from "react";
const pages = [
  "config",
  "generate",
  "review",
  "versions",
  "train",
  "eval",
] as const;
type Page = (typeof pages)[number];
type Route = { page: Page; projectId: string; versionId: string };
function read(): Route {
  const [path, search] = window.location.hash.slice(1).split("?");
  const page = (path || "").replace(/^\//, "") as Page;
  const params = new URLSearchParams(search);
  return {
    page: pages.includes(page) ? page : "config",
    projectId: params.get("project") || "",
    versionId: params.get("version") || "",
  };
}
export function useWorkspaceRoute() {
  const [route, setRoute] = useState<Route>(read);
  useEffect(() => {
    const sync = () => setRoute(read());
    window.addEventListener("popstate", sync);
    window.addEventListener("hashchange", sync);
    return () => {
      window.removeEventListener("popstate", sync);
      window.removeEventListener("hashchange", sync);
    };
  }, []);
  useEffect(() => {
    const query = new URLSearchParams();
    if (route.projectId) query.set("project", route.projectId);
    if (route.versionId) query.set("version", route.versionId);
    const hash = `#/${route.page}${query.size ? `?${query}` : ""}`;
    if (window.location.hash !== hash) {
      if (!window.location.hash || read().page === route.page)
        window.history.replaceState(null, "", hash);
      else window.history.pushState(null, "", hash);
    }
  }, [route]);
  return {
    ...route,
    setPage: (page: Page) => setRoute((old) => ({ ...old, page })),
    setProjectId: (projectId: string) =>
      setRoute((old) => ({
        ...old,
        projectId,
        versionId: projectId === old.projectId ? old.versionId : "",
      })),
    setVersionId: (versionId: string) =>
      setRoute((old) => ({ ...old, versionId })),
  };
}
