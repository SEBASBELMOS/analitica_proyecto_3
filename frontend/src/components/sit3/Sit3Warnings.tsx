// Acordeon de advertencias metodologicas (siempre visible pero no invasivo).
import { useState } from "react";

interface Props {
  warnings: string[];
}

export default function Sit3Warnings({ warnings }: Props) {
  const [open, setOpen] = useState(false);
  if (!warnings.length) return null;

  return (
    <div className="border-t border-slate-200 dark:border-slate-700 bg-amber-50 dark:bg-amber-900/10">
      <button
        onClick={() => setOpen((v) => !v)}
        className="w-full px-4 py-2 flex items-center justify-between text-left hover:bg-amber-100 dark:bg-amber-900/30 transition-colors"
      >
        <span className="text-xs font-bold text-amber-900 dark:text-amber-400 flex items-center gap-2">
          <svg
            xmlns="http://www.w3.org/2000/svg"
            className="w-4 h-4"
            viewBox="0 0 20 20"
            fill="currentColor"
          >
            <path
              fillRule="evenodd"
              d="M8.485 2.495c.673-1.167 2.357-1.167 3.03 0l6.28 10.875c.673 1.167-.17 2.625-1.516 2.625H3.72c-1.347 0-2.189-1.458-1.515-2.625L8.485 2.495zM10 6a.75.75 0 01.75.75v3.5a.75.75 0 01-1.5 0v-3.5A.75.75 0 0110 6zm0 9a1 1 0 100-2 1 1 0 000 2z"
              clipRule="evenodd"
            />
          </svg>
          Advertencias metodologicas ({warnings.length})
        </span>
        <span className="text-xs text-amber-700 dark:text-amber-400">{open ? "Ocultar" : "Mostrar"}</span>
      </button>
      {open && (
        <ul className="px-4 pb-3 space-y-1.5">
          {warnings.map((w, i) => (
            <li key={i} className="text-xs text-amber-900 dark:text-amber-400 flex gap-2">
              <span className="text-amber-600 dark:text-amber-500 font-bold">·</span>
              <span>{w}</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
