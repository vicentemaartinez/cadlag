import Link from "next/link";
export default function NotFound() {
  return (<section className="wrap" style={{ paddingBlock: 120 }}><h1 style={{ fontSize: 48 }}>Esta página no existe</h1><p style={{ marginTop: 12, color: "var(--ink-2)" }}><Link href="/">Volver a la portada</Link></p></section>);
}
