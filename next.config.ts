import type { NextConfig } from 'next';

const nextConfig: NextConfig = {
  experimental: {
    ppr: false,
    clientSegmentCache: false,
  }
};

export default nextConfig;

module.exports = {
  distDir: 'out',
  output: "export",
}
