import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";

// IMPORTANTE: leaflet.css ANTES de cualquier import propio para garantizar
// que el CSS de tiles este cargado antes que React renderice los componentes.
import "leaflet/dist/leaflet.css";
import "./styles/index.css";

// Fix de iconos Leaflet con bundlers (Vite/Webpack rompen las URLs default).
import L from "leaflet";
import iconUrl from "leaflet/dist/images/marker-icon.png";
import iconRetinaUrl from "leaflet/dist/images/marker-icon-2x.png";
import shadowUrl from "leaflet/dist/images/marker-shadow.png";

// Override de iconos por defecto de Leaflet (propiedad privada _getIconUrl).
// El cast `as any` evita el error TS sobre propiedad privada.
delete (L.Icon.Default.prototype as any)._getIconUrl;
L.Icon.Default.mergeOptions({ iconUrl, iconRetinaUrl, shadowUrl });

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
