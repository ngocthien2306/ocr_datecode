import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'


export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    allowedHosts: ['vlm-agent.ngrok.dev', "factory.ngrok.app"], 
    proxy: {
      '/api': {
        target: 'https://vlm-agent-api.ngrok.dev',
        changeOrigin: true,
        configure: (proxy) => {
          proxy.on('proxyRes', function (proxyRes, req) {
            // Disable buffering for SSE/streaming endpoints
            if (req.url.includes('/stream')) {
              proxyRes.headers['x-accel-buffering'] = 'no';
            }
          });
        }
      }
    }
  }
})