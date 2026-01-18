import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';

export default defineConfig(({ mode }) => {
  // Load env file based on `mode` in the current working directory.
  // Set the third parameter to '' to load all env regardless of the `VITE_` prefix.
  const env = loadEnv(mode, process.cwd(), '');

  return {
    plugins: [react()],
    resolve: {
      alias: {
        '@': path.resolve(__dirname, './src'),
      },
    },
    server: {
      port: parseInt(env.VITE_DEV_SERVER_PORT || '5173'),
      host: env.VITE_DEV_SERVER_HOST === 'true',
      allowedHosts: [
        'suntech-vision.ngrok.app',
        'localhost',
        '.loca.lt'
      ],
    },
    build: {
      outDir: 'dist',
      sourcemap: true,
    },
  };
});
