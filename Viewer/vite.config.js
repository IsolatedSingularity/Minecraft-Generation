import { defineConfig } from "vite"
import vue from "@vitejs/plugin-vue"
import { resolve } from "node:path"

export default defineConfig({
  plugins: [vue()],
  build: {
    rollupOptions: {
      input: {
        structures: resolve(__dirname, "index.html"),
        seedMap: resolve(__dirname, "seed-map.html")
      }
    }
  }
})
