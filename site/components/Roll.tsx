"use client";
import { useEffect, useRef } from "react";

// Número cuyas cifras ruedan de su valor anterior al nuevo (solo si la forma del texto no cambia).
function html(str: string) {
  return [...str].map((ch) => /\d/.test(ch)
    ? `<span class="d"><span class="col" style="transform:translateY(-${ch}em)">${[...Array(10).keys()].map((i) => `<span>${i}</span>`).join("")}</span></span>`
    : `<span class="sep">${ch}</span>`).join("");
}
export default function Roll({ value, className, suffix }: { value: string; className?: string; suffix?: string }) {
  const el = useRef<HTMLSpanElement>(null); const prev = useRef<string | null>(null);
  useEffect(() => {
    const e = el.current!; const p = prev.current; prev.current = value;
    const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (!p || p.replace(/\d/g, "0") !== value.replace(/\d/g, "0") || reduced) { e.innerHTML = html(value); return; }
    e.innerHTML = html(p);
    requestAnimationFrame(() => requestAnimationFrame(() => {
      const cols = e.querySelectorAll<HTMLElement>(".col"); let j = 0;
      [...value].forEach((ch) => { if (/\d/.test(ch)) { cols[j].style.transitionDelay = j * 45 + "ms"; cols[j].style.transform = `translateY(-${ch}em)`; j++; } });
    }));
  }, [value]);
  return <span className={className}><span className="roll" ref={el}>{value}</span>{suffix && <small>{suffix}</small>}</span>;
}
