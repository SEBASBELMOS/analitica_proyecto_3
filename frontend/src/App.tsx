// Layout principal: header + mapa Situacion 3.
import { useEffect, useState } from "react";

import { API_BASE_URL, getHealth } from "./api/client";
import ThemeToggle from "./components/ThemeToggle";
import Situacion3Page from "./pages/Situacion3Page";
import type { HealthResponse } from "./types/api";

export default function App() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthError, setHealthError] = useState<string | null>(null);

  useEffect(() => {
    getHealth()
      .then(setHealth)
      .catch((e) => setHealthError(e?.message ?? "Error desconocido"));
  }, []);

  return (
    <div className="flex flex-col h-screen bg-slate-50 dark:bg-slate-950 text-slate-900 dark:text-slate-100 transition-colors">
      {/* Header */}
      <header className="bg-slate-900 text-white shadow-lg px-6 py-3 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-lg bg-indigo-600 flex items-center justify-center font-bold text-lg">
            G
          </div>
          <div>
            <h1 className="text-base font-bold leading-tight">GeoVision-CLIP Cali</h1>
            <p className="text-[11px] opacity-70 leading-tight">
              Estimacion espacio-temporal de NO₂, SO₂, O₃ - UAO 2026
            </p>
          </div>
        </div>

        <div className="flex items-center gap-3">
          <div className="text-right text-[11px]">
            {health ? (
              <div className="flex flex-col items-end gap-0.5">
                <div className="flex items-center gap-1.5">
                  <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
                  <span>
                    API v{health.version} - {health.n_grid_cells} celdas
                  </span>
                </div>
                <div className="opacity-60 font-mono">
                  {API_BASE_URL.replace(/^https?:\/\//, "")}
                </div>
              </div>
            ) : healthError ? (
              <div className="flex items-center gap-1.5 text-rose-300">
                <span className="w-2 h-2 rounded-full bg-rose-400"></span>
                <span>API offline</span>
              </div>
            ) : (
              <span className="opacity-60">Conectando...</span>
            )}
          </div>
          <ThemeToggle />
        </div>
      </header>

      {/* Contenido */}
      <main className="flex-1 overflow-hidden">
        <Situacion3Page />
      </main>

      {/* Footer */}
      <footer className="bg-slate-900 text-slate-400 text-[11px] px-6 py-2 flex items-center justify-between border-t border-slate-700">
        <div>
          Proyecto Analitica de Datos I - Universidad Autonoma de Occidente
        </div>
        <div className="flex gap-4">
          <span>Sebastian Belalcazar</span>
          <span>Manuel Gruezo</span>
          <span>Luis Garcia</span>
        </div>
      </footer>
    </div>
  );
}
