import type { Metadata, Viewport } from "next";
import Link from "next/link";
import "./globals.css";
import Nav from "@/components/Nav";
import Grain from "@/components/Grain";
import Glyph from "@/components/Glyph";
import Sw from "@/components/Sw";
import { getGlobal, getIndex } from "@/lib/data";
import { BASE } from "@/lib/leagues";

export const metadata: Metadata = {
  title: { default: "càdlàg", template: "%s · càdlàg" },
  description: "Probabilidades de cómo termina cada liga: nueve ligas simuladas 20.000 veces con sus reglas reales de desempate.",
  icons: { icon: `${BASE}/icon.svg`, apple: `${BASE}/apple-touch-icon.png` },
  appleWebApp: { capable: true, title: "càdlàg", statusBarStyle: "default" },
};
export const viewport: Viewport = { width: "device-width", initialScale: 1, viewportFit: "cover", themeColor: "#ECE7DD" };

// La web parte en claro; el oscuro solo si la persona lo eligió antes.
const themeScript = `(function(){var d=document.documentElement;try{var k=localStorage.getItem('cadlag-theme')==='dark';d.setAttribute('data-theme',k?'dark':'light');if(k)document.addEventListener('DOMContentLoaded',function(){var m=document.querySelector('meta[name=theme-color]');if(m)m.setAttribute('content','#2A2622')})}catch(e){d.setAttribute('data-theme','light')}try{if(sessionStorage.getItem('cadlag-seen'))d.classList.add('seen')}catch(e){}})()`;

export default function RootLayout({ children }: { children: React.ReactNode }) {
  const idx = getIndex(); const G = getGlobal();
  const summary = idx.leagues.map((l) => { const v = (t: (typeof l.teams)[number]) => t.asc ?? t.title ?? t.p[0]; let b = l.teams[0]; for (const t of l.teams) if (v(t) > v(b)) b = t; return { comp: l.comp, fav: b.name, p: v(b) }; });
  const gen = new Date().toLocaleDateString("es-CL", { day: "numeric", month: "long", year: "numeric" });
  return (
    <html lang="es" data-theme="light" suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: themeScript }} />
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="" />
        {/* eslint-disable-next-line @next/next/no-page-custom-font */}
        <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wdth,wght@62..125,100..900&family=Zen+Old+Mincho:wght@400;600&display=swap" />
      </head>
      <body>
        <Sw />
        <Grain />
        <Nav summary={summary} topTeam={G.ranking[0].name} lambda2={G.eigen[1]} />
        <main>{children}</main>
        <footer><div className="wrap">
          <Link className="brand" href="/"><Glyph />càdlàg</Link>
          <span>Datos de ESPN · cuotas de football-data.co.uk · generado el {gen}</span>
        </div></footer>
      </body>
    </html>
  );
}
