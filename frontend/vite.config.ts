import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import path from "path";

export default defineConfig(({ mode }) => ({
  base: "/",
  plugins: [
    react(),
    tailwindcss(),
  ],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "src"),
    },
    dedupe: ["react", "react-dom"],
  },
  server: {
    port: 3000,
    open: true,
    // Dev proxy: forwards /api/* to backend so no CORS issues in development
    proxy: {
      "/api": {
        target: process.env.VITE_API_URL || "http://localhost:3001",
        changeOrigin: true,
        secure: false,
      },
      "/uploads": {
        target: process.env.VITE_API_URL || "http://localhost:3001",
        changeOrigin: true,
        secure: false,
      },
    },
  },
  build: {
    outDir: "dist",
    emptyOutDir: true,
    // Chunk splitting for better caching
    rollupOptions: {
      output: {
        manualChunks: {
          react:    ["react", "react-dom"],
          router:   ["wouter"],
          ui:       ["@radix-ui/react-dialog", "@radix-ui/react-dropdown-menu", "@radix-ui/react-tooltip"],
          charts:   ["recharts"],
          motion:   ["framer-motion"],
          i18n:     ["i18next", "react-i18next"],
        },
      },
    },
  },
}));
