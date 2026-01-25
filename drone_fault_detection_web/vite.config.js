import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: { // CORS güvenlik hatalarını aşmak için proxy ayarları
      '/api': {
        target: 'http://localhost:3001',
        changeOrigin: true
      }
    }
  }
})
