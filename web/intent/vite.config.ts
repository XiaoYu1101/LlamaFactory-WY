import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  base: "/intent/",
  plugins: [react()],
  build: { outDir: "../../src/llamafactory/intent/static", emptyOutDir: true },
  server: { proxy: { "/intent-api": "http://127.0.0.1:7861" } },
});
