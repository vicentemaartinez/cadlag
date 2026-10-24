export const fmt = (x: number, d = 1) => x.toLocaleString("es-CL", { minimumFractionDigits: d, maximumFractionDigits: d });
export const fmtInt = (x: number) => Math.round(x).toLocaleString("es-CL");
export const pct = (p: number, d = 1) => fmt(p * 100, d);

const STOP = new Set(["de", "del", "la", "el", "fc", "cd", "afc", "club", "sc", "ac", "as", "rc", "cf", "the", "and", "&"]);
export const initials = (n: string) => {
  const w = n.replace(/[^\p{L}\s]/gu, "").split(/\s+/).filter((x) => x && !STOP.has(x.toLowerCase()));
  return (w.length > 1 ? w[0][0] + w[1][0] : (w[0] || n).slice(0, 2)).toUpperCase();
};

export const dateShort = (s: string, opts: Intl.DateTimeFormatOptions = { day: "numeric", month: "short" }) =>
  new Date(s.includes("T") || s.includes("+") ? s.replace(" ", "T") : s.replace(" ", "T") + "Z").toLocaleDateString("es-CL", opts);

export const rounds = (played: number, total: number, n: number) => [Math.round(played / (n / 2)), total / (n / 2)] as const;
