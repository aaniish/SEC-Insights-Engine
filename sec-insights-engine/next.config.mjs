/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'standalone',
  eslint: {
    ignoreDuringBuilds: true,
  },
  typescript: {
    ignoreBuildErrors: true,
  },
  images: {
    unoptimized: true,
  },
  experimental: {
    webpackBuildWorker: true,
    parallelServerBuildTraces: true,
    parallelServerCompiles: true,
  },

  // Add rewrites for API proxying during development
  async rewrites() {
    return [
      {
        source: '/api/:path*', // Match any path starting with /api/
        // Use the backend service name from docker-compose for container-to-container communication
        destination: 'http://backend:8000/api/:path*', 
      },
    ]
  },
}

export default nextConfig
