/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // Bake NEXT_PUBLIC_ vars at build time with fallbacks for Docker builds
  // where ARG/ENV may not have been propagated due to layer caching.
  env: {
    NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000',
    NEXT_PUBLIC_API_KEY: process.env.NEXT_PUBLIC_API_KEY || 'flowwatch-dev-key-001',
  },
};

module.exports = nextConfig;
