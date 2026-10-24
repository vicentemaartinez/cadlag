"use client";
import Link from "next/link";
import { useEffect, useRef } from "react";
import { ABBR, NAMES, leagueHref } from "@/lib/leagues";

// Barra de ligas (vidrio) con indicador que se desliza a la pestaña activa.
export default function LeagueTabs({ current, leagues }: { current: string; leagues: string[] }) {
  const host = useRef<HTMLDivElement>(null);
  const ind = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const move = () => {
      const a = host.current?.querySelector<HTMLElement>("[aria-current='page']"); if (!a || !ind.current) return;
      ind.current.style.width = a.offsetWidth + "px"; ind.current.style.transform = `translateX(${a.offsetLeft}px)`;
      const h = host.current!; if (a.offsetLeft < h.scrollLeft || a.offsetLeft + a.offsetWidth > h.scrollLeft + h.clientWidth) h.scrollLeft = a.offsetLeft - 16;
    };
    move(); document.fonts?.ready.then(move); addEventListener("resize", move);
    return () => removeEventListener("resize", move);
  }, [current]);
  return (
    <div className="lbar"><div className="wrap"><div className="tabs" ref={host}>
      <div className="tab-ind" ref={ind} />
      {leagues.map((c) => (
        <Link key={c} className="tab" href={leagueHref(c)} aria-current={c === current ? "page" : undefined}>
          <span className="abbr">{ABBR[c]}</span><span className="full">{NAMES[c]}</span>
        </Link>
      ))}
      <Link className="tab sep" href="/mundo/" aria-current={current === "mundo" ? "page" : undefined}>Mundo</Link>
    </div></div></div>
  );
}
