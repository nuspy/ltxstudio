import { useState } from 'react';
import { Box, Button, Card, CardContent, CircularProgress, Typography } from '@mui/material';
import VerifiedIcon from '@mui/icons-material/Verified';
import CancelIcon from '@mui/icons-material/Cancel';
import { detect, type Detection } from '../detect/stego';

export default function Detector() {
  const [busy, setBusy] = useState(false);
  const [res, setRes] = useState<Detection | null>(null);
  const [err, setErr] = useState('');

  const pick = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (!f) return;
    setBusy(true);
    setRes(null);
    setErr('');
    const img = new Image();
    img.onload = async () => {
      try {
        const c = document.createElement('canvas');
        const scale = Math.min(1, 1600 / Math.max(img.width, img.height));
        c.width = Math.round(img.width * scale);
        c.height = Math.round(img.height * scale);
        const ctx = c.getContext('2d')!;
        ctx.drawImage(img, 0, 0, c.width, c.height);
        const id = ctx.getImageData(0, 0, c.width, c.height);
        setRes(await detect(id));
      } catch (er: any) {
        setErr(er?.message || String(er));
      } finally {
        setBusy(false);
      }
    };
    img.onerror = () => { setErr('Immagine non valida'); setBusy(false); };
    const rd = new FileReader();
    rd.onload = () => { img.src = rd.result as string; };
    rd.readAsDataURL(f);
  };

  return (
    <Box sx={{ p: 2 }}>
      <Typography variant="h6" sx={{ mb: 1 }}>Rilevatore artefatto</Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
        Verifica se una foto è stata creata con questa app (firma invisibile). Analisi in locale sul dispositivo,
        senza rete.
      </Typography>
      <Button component="label" variant="outlined">
        Scegli foto
        <input hidden type="file" accept="image/*" onChange={pick} />
      </Button>

      {busy && (
        <Box sx={{ textAlign: 'center', mt: 3 }}>
          <CircularProgress />
          <Typography variant="body2" sx={{ mt: 1 }}>Analizzo…</Typography>
        </Box>
      )}
      {err && <Typography color="error" sx={{ mt: 2 }}>{err}</Typography>}
      {res && (
        <Card sx={{ mt: 2 }}>
          <CardContent sx={{ textAlign: 'center' }}>
            {res.found ? (
              <>
                <VerifiedIcon sx={{ fontSize: 44, color: 'success.main' }} />
                <Typography variant="h6" color="success.main">Creato con questa app</Typography>
                {res.text && (
                  <Typography variant="body2" sx={{ mt: 1 }}>
                    Testo firma: <b>&quot;{res.text}&quot;</b>
                  </Typography>
                )}
                <Typography variant="caption" color="text.secondary">
                  affidabilità {res.z.toFixed(1)}
                </Typography>
              </>
            ) : (
              <>
                <CancelIcon sx={{ fontSize: 44, color: 'text.disabled' }} />
                <Typography variant="h6" color="text.secondary">Nessuna firma rilevata</Typography>
                <Typography variant="caption" color="text.secondary">
                  Non risulta creato con questa app (o foto molto ruotata/rovinata).
                </Typography>
              </>
            )}
          </CardContent>
        </Card>
      )}
    </Box>
  );
}
