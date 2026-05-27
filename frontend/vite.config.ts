import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Vite config para GeoVision-CLIP Cali
// HF Spaces Docker requiere servir en puerto 7860 (esto solo aplica a `preview`/runtime)
// `allowedHosts` debe listar explicitamente los hosts permitidos para preview.
// HF Spaces usa el patron {owner}-{space}.hf.space, agregamos wildcard .hf.space
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    host: true,
  },
  preview: {
    port: 7860,
    host: true,
    allowedHosts: [
      "analiticalastdance-geovision-cali-frontend.hf.space",        
      "localhost",
      "127.0.0.1",
    ],
  },
  build: {
    outDir: "dist",
    sourcemap: false,
    chunkSizeWarningLimit: 1000,
  },
});
