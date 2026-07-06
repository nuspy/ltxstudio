import type { CapacitorConfig } from '@capacitor/cli';

// L'appId/nome può essere sovrascritto per edizione dal builder feature_matrix (vedi S7).
const config: CapacitorConfig = {
  appId: 'com.aitoolsforge.ltxstudio',
  appName: 'LTX Studio',
  webDir: 'dist',
  server: {
    androidScheme: 'https',
  },
  android: {
    allowMixedContent: true, // consente http verso il server locale/LAN dell'utente
  },
};

export default config;
