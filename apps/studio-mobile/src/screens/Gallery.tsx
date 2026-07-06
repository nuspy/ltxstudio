import { useEffect, useState } from 'react';
import { Box, Button, Grid, Typography } from '@mui/material';
import DownloadIcon from '@mui/icons-material/Download';
import { clearResults, loadResults } from '../gallery';
import type { JobFile } from '../api';
import { saveToDevice } from '../download';

const isVideo = (n: string) => /\.(mp4|webm|mkv|mov)$/i.test(n);

export default function Gallery() {
  const [items, setItems] = useState<JobFile[]>([]);
  useEffect(() => {
    loadResults().then(setItems);
  }, []);

  if (!items.length) {
    return (
      <Box sx={{ p: 4, textAlign: 'center', color: 'text.secondary' }}>
        <Typography>Nessun risultato ancora. Genera qualcosa!</Typography>
      </Box>
    );
  }

  return (
    <Box sx={{ p: 2 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
        <Typography variant="h6">Galleria</Typography>
        <Button size="small" color="inherit" onClick={() => { clearResults(); setItems([]); }}>
          Svuota
        </Button>
      </Box>
      <Grid container spacing={1}>
        {items.map((f) => (
          <Grid item xs={6} key={f.path}>
            {isVideo(f.name) ? (
              <video src={f.download_url} controls style={{ width: '100%', borderRadius: 8 }} />
            ) : (
              <img src={f.download_url} alt={f.name} style={{ width: '100%', borderRadius: 8 }} />
            )}
            <Button fullWidth size="small" startIcon={<DownloadIcon />} onClick={() => saveToDevice(f.download_url, f.name)}>
              Scarica
            </Button>
          </Grid>
        ))}
      </Grid>
    </Box>
  );
}
