import type { MetadataRoute } from "next";
import { BASE } from "@/lib/leagues";

export const dynamic = "force-static";

// Para "Agregar a pantalla de inicio": abre sin barra del navegador, con el ícono de càdlàg.
export default function manifest(): MetadataRoute.Manifest {
  return {
    name: "càdlàg",
    short_name: "càdlàg",
    description: "Probabilidades de cómo termina cada liga.",
    lang: "es",
    start_url: `${BASE}/`,
    scope: `${BASE}/`,
    display: "standalone",
    background_color: "#ECE7DD",
    theme_color: "#ECE7DD",
    icons: [
      { src: `${BASE}/icon-192.png`, sizes: "192x192", type: "image/png" },
      { src: `${BASE}/icon-512.png`, sizes: "512x512", type: "image/png" },
      { src: `${BASE}/icon-maskable-512.png`, sizes: "512x512", type: "image/png", purpose: "maskable" },
    ],
  };
}
