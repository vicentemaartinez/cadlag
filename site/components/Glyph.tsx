// El salto càdlàg: tramo con círculo abierto al final, tramo nuevo con punto lleno al inicio.
export default function Glyph({ className = "glyph" }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 22 16" aria-hidden="true">
      <line x1="2" y1="12" x2="10" y2="12" />
      <circle className="o" cx="11.5" cy="12" r="2" />
      <line x1="12" y1="4" x2="20" y2="4" />
      <circle className="c" cx="12" cy="4" r="2" />
    </svg>
  );
}
