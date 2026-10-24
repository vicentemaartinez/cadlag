// Rampa de calor en OKLab entre --heat-0 y --heat-1 (se relee al cambiar de tema).
function hexRGB(h: string) { h = h.replace("#", ""); return [0, 2, 4].map((i) => parseInt(h.slice(i, i + 2), 16) / 255); }
const lin = (c: number) => (c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4));
const gam = (c: number) => (c <= 0.0031308 ? 12.92 * c : 1.055 * Math.pow(c, 1 / 2.4) - 0.055);
function toLab(hex: string) {
  const [r, g, b] = hexRGB(hex).map(lin);
  const l = Math.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b);
  const m = Math.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b);
  const s = Math.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b);
  return [0.2104542553 * l + 0.793617785 * m - 0.0040720468 * s, 1.9779984951 * l - 2.428592205 * m + 0.4505937099 * s, 0.0259040371 * l + 0.7827717662 * m - 0.808675766 * s];
}
function fromLab([L, A, B]: number[]) {
  const l = (L + 0.3963377774 * A + 0.2158037573 * B) ** 3, m = (L - 0.1055613458 * A - 0.0638541728 * B) ** 3, s = (L - 0.0894841775 * A - 1.291485548 * B) ** 3;
  const rgb = [4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s, -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s, -0.0041960863 * l - 0.7034186147 * m + 1.707614701 * s];
  return "rgb(" + rgb.map((v) => Math.round(Math.max(0, Math.min(1, gam(v))) * 255)).join(",") + ")";
}

let A: number[] = toLab("#F1DED4"), Bc: number[] = toLab("#7E2128");
export function reloadHeat() {
  if (typeof window === "undefined") return;
  const cs = getComputedStyle(document.documentElement);
  A = toLab(cs.getPropertyValue("--heat-0").trim() || "#F1DED4");
  Bc = toLab(cs.getPropertyValue("--heat-1").trim() || "#7E2128");
}
export function heat(p: number) {
  const t = Math.pow(Math.min(1, p / 0.6), 0.62);
  return fromLab(A.map((v, i) => v + (Bc[i] - v) * t));
}
export const cssVar = (name: string) => (typeof window === "undefined" ? "" : getComputedStyle(document.documentElement).getPropertyValue(name).trim());

// Cambios de tema: un evento propio para que cada componente repinte lo que pinta a mano.
export const THEME_EVENT = "cadlag-theme";
export function onTheme(fn: () => void) {
  const h = () => { reloadHeat(); fn(); };
  window.addEventListener(THEME_EVENT, h);
  return () => window.removeEventListener(THEME_EVENT, h);
}
