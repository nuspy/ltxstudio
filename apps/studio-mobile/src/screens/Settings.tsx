import { useEffect, useState } from 'react';
import { Alert, Box, Button, Stack, TextField, Typography } from '@mui/material';
import { AppSettings, DEFAULTS, loadSettings, saveSettings } from '../settings';
import { fetchFeatures, health, setApi } from '../api';
import { setServerSnapshot } from '../features';

export default function Settings() {
  const [s, setS] = useState<AppSettings>(DEFAULTS);
  const [status, setStatus] = useState('');

  useEffect(() => {
    loadSettings().then(setS);
  }, []);

  const save = async () => {
    await saveSettings(s);
    setApi(s.serverUrl, s.token);
    setStatus('Salvato.');
  };

  const test = async () => {
    setApi(s.serverUrl, s.token);
    try {
      const h = await health();
      const f = await fetchFeatures();
      setServerSnapshot(f);
      setStatus(`OK — backend ${h.native_backend_up ? 'attivo' : 'spento'}, edizione ${f.edition}`);
    } catch (e: any) {
      setStatus('Errore: ' + e.message);
    }
  };

  return (
    <Box sx={{ p: 2 }}>
      <Typography variant="h6" sx={{ mb: 2 }}>Impostazioni</Typography>
      <Stack spacing={2}>
        <TextField
          label="URL server"
          value={s.serverUrl}
          onChange={(e) => setS({ ...s, serverUrl: e.target.value })}
          helperText="es. http://192.168.1.50:8200  oppure  https://fx.tadamoo.com"
        />
        <TextField
          label="Token (opzionale, per server remoto)"
          value={s.token}
          onChange={(e) => setS({ ...s, token: e.target.value })}
        />
        <Box>
          <Button onClick={save}>Salva</Button>
          <Button variant="outlined" onClick={test} sx={{ ml: 1 }}>Test connessione</Button>
        </Box>
        {status && <Alert severity={status.startsWith('Errore') ? 'error' : 'success'}>{status}</Alert>}
        <Typography variant="caption" color="text.secondary">
          Il server deve essere avviato in ascolto sulla rete (LTX_API_HOST=0.0.0.0). Per l'accesso da
          Internet (tadamoo.com) usa HTTPS + token dietro un reverse-proxy.
        </Typography>
      </Stack>
    </Box>
  );
}
