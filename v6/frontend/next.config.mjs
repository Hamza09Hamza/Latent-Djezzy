/** @type {import('next').NextConfig} */
const nextConfig = {
  eslint: {
    ignoreDuringBuilds: true,
  },
  typescript: {
    ignoreBuildErrors: true,
  },
  images: {
    // chart PNGs are served from the ngrok tunnel, not optimized by Next
    unoptimized: true,
  },
}

export default nextConfig
