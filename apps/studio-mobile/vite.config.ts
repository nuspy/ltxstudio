import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// base './' → gli asset funzionano dentro il WebView Capacitor (file://) e come PWA.
export default defineConfig({
  plugins: [react()],
  base: './',
  server: { port: 5173, host: true },
});
