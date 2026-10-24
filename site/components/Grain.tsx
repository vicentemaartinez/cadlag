"use client";
import { useEffect, useRef } from "react";
import { onTheme } from "@/lib/heat";

// Fibra de papel: ruido más unas pocas fibras largas, generado una vez como imagen de fondo.
export default function Grain() {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const draw = () => {
      const c = document.createElement("canvas"), s = 220; c.width = s; c.height = s;
      const x = c.getContext("2d")!; const d = x.createImageData(s, s);
      for (let i = 0; i < d.data.length; i += 4) { const v = (Math.random() * 255) | 0; d.data[i] = d.data[i + 1] = d.data[i + 2] = v; d.data[i + 3] = 255; }
      x.putImageData(d, 0, 0); x.globalAlpha = 0.25; x.strokeStyle = "#000"; x.lineWidth = 0.5;
      for (let i = 0; i < 26; i++) {
        x.beginPath(); const x0 = Math.random() * s, y0 = Math.random() * s, a = Math.random() * Math.PI;
        x.moveTo(x0, y0); x.quadraticCurveTo(x0 + Math.cos(a) * 18 + Math.random() * 8, y0 + Math.sin(a) * 18, x0 + Math.cos(a) * 36, y0 + Math.sin(a) * 36); x.stroke();
      }
      if (ref.current) ref.current.style.backgroundImage = `url(${c.toDataURL()})`;
    };
    draw(); return onTheme(draw);
  }, []);
  return <div className="grain" ref={ref} aria-hidden="true" />;
}
