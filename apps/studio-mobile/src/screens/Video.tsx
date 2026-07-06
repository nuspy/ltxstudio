import { useState } from 'react';
import { Box, Button, MenuItem, Stack, TextField, Typography } from '@mui/material';
import { FeatureGate } from '../components/FeatureGate';
import { ModelPicker, type Components } from '../components/ModelPicker';
import { JobPanel } from '../components/JobPanel';
import { useJob } from '../hooks/useJob';
import { uploadFile } from '../api';
import { isEnabled } from '../features';

type Mode = 't2v' | 'i2v' | 'controlnet';

export default function Video() {
  const [mode, setMode] = useState<Mode>('t2v');
  const [model, setModel] = useState('');
  const [comp, setComp] = useState<Components>({});
  const [prompt, setPrompt] = useState('');
  const [imgPath, setImgPath] = useState('');
  const [ctrlPath, setCtrlPath] = useState('');
  const [ctrlType, setCtrlType] = useState('canny');
  const [len, setLen] = useState(3);
  const [staging, setStaging] = useState(false);
  const { job, busy, error, run } = useJob();

  const pick = async (e: React.ChangeEvent<HTMLInputElement>, set: (p: string) => void) => {
    const f = e.target.files?.[0];
    if (!f) return;
    setStaging(true);
    try {
      const up = await uploadFile(f, f.name);
      set(up.path);
    } finally {
      setStaging(false);
    }
  };

  const gen = () => {
    if (mode === 'controlnet') {
      return run('/video/controlnet', { prompt, control_video: ctrlPath, control_type: ctrlType, length_s: len });
    }
    return run('/video/generate', {
      prompt, model, length_s: len, steps: comp.steps ?? 20,
      ...(mode === 'i2v' ? { image_path: imgPath } : {}),
    });
  };

  const canGen = !busy && !staging && prompt &&
    (mode === 'controlnet' ? !!ctrlPath : (!!model && (mode === 't2v' || !!imgPath)));

  return (
    <FeatureGate cap="cap.video">
      <Box sx={{ p: 2 }}>
        <Typography variant="h6" sx={{ mb: 2 }}>Video</Typography>
        <TextField select fullWidth label="Modalità" value={mode} onChange={(e) => setMode(e.target.value as Mode)}>
          <MenuItem value="t2v">Testo → Video</MenuItem>
          <MenuItem value="i2v">Immagine → Video</MenuItem>
          {isEnabled('cap.controlnet') && <MenuItem value="controlnet">ControlNet</MenuItem>}
        </TextField>

        {mode !== 'controlnet' && (
          <Box sx={{ mt: 2 }}>
            <ModelPicker task="video" value={model} onChange={setModel} onComponents={setComp} />
          </Box>
        )}

        {mode === 'i2v' && (
          <Button component="label" variant="outlined" sx={{ mt: 2 }}>
            {imgPath ? 'Immagine caricata ✓' : 'Immagine di partenza'}
            <input hidden type="file" accept="image/*" onChange={(e) => pick(e, setImgPath)} />
          </Button>
        )}

        {mode === 'controlnet' && (
          <Stack spacing={2} sx={{ mt: 2 }}>
            <Button component="label" variant="outlined">
              {ctrlPath ? 'Video guida caricato ✓' : 'Video guida (ControlNet)'}
              <input hidden type="file" accept="video/*" onChange={(e) => pick(e, setCtrlPath)} />
            </Button>
            <TextField select label="Tipo controllo" value={ctrlType} onChange={(e) => setCtrlType(e.target.value)}>
              <MenuItem value="canny">Canny (contorni)</MenuItem>
              <MenuItem value="depth">Depth (profondità)</MenuItem>
              <MenuItem value="pose">Pose (posa)</MenuItem>
            </TextField>
          </Stack>
        )}

        <TextField label="Descrizione (prompt)" fullWidth multiline minRows={2}
          value={prompt} onChange={(e) => setPrompt(e.target.value)} sx={{ my: 2 }} />
        <TextField type="number" label="Durata (s)" value={len}
          onChange={(e) => setLen(Math.max(1, Number(e.target.value) || 1))} sx={{ mb: 2 }} />

        <Button fullWidth disabled={!canGen} onClick={gen}>
          {staging ? 'Carico…' : busy ? 'Genero…' : 'Genera video'}
        </Button>
        <JobPanel job={job} busy={busy} error={error} />
      </Box>
    </FeatureGate>
  );
}
