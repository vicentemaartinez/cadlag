import type { NextConfig } from "next";

// Sitio estático (next build → out/). Para GitHub Pages bajo /cadlag: NEXT_PUBLIC_BASE_PATH=/cadlag
const basePath = process.env.NEXT_PUBLIC_BASE_PATH || "";

const config: NextConfig = {
  output: "export",
  trailingSlash: true,
  basePath,
  images: { unoptimized: true },
  reactStrictMode: true,
};

export default config;
