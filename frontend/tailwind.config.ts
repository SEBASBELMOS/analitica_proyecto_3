import type { Config } from "tailwindcss";

export default {
  //corePlugins: {
    //preflight: false,  // desactiva el reset CSS
  //},
  darkMode: "class",
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        // Colores paleta GeoVision
        primary: "#2c7bb6",       // azul Cali
        accent: "#d7191c",        // rojo hot spots
        cold: "#abd9e9",          // azul claro cold
        warning: "#fdae61",       // naranja
        neutral: "#cccccc",       // gris no significativo
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "monospace"],
      },
    },
  },
  plugins: [],
} satisfies Config;
