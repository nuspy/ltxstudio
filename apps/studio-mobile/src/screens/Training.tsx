import { useState } from 'react';
import { Box, Button, TextField, Typography } from '@mui/material';
import { FeatureGate } from '../components/FeatureGate';
import { JobPanel } from '../components/JobPanel';
import { useJob } from '../hooks/useJob';
import { uploadFile } from '../api';

export default function Training() {
  const [name, setName] = useState('');
  const [trigger, setTrigger] = useState('');
  const [steps, setSteps] = useState(2000);
  const [images, setImages] = useState<{ path: string; caption: string }[]>([]);
  const [staging, setStaging] = useState(false);
  const { job, busy, error, run } = useJob();

  const add = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const fs = e.target.files;
    if (!fs) return;
    setStaging(true);
    try {
      for (const f of Array.from(fs)) {
        const up = await uploadFile(f, f.name);
        setImages((prev) => [...prev, { path: up.path, caption: trigger }]);
      }
    } finally {
      setStaging(false);
    }
  };

  const train = () =>
    run('/train/lora_image', {
      name: name || 'my_lora',
      trigger,
      steps,
      images: images.map((i) => ({ path: i.path, caption: i.caption })),
    });

  return (
    <FeatureGate cap="cap.lora_image_train">
      <Box sx={{ p: 2 }}>
        <Typography variant="h6" sx={{ mb: 2 }}>Allena LoRA (immagini)</Typography>
        <TextField label="Nome LoRA" fullWidth value={name} onChange={(e) => setName(e.target.value)} />
        <TextField label="Trigger word" fullWidth value={trigger} onChange={(e) => setTrigger(e.target.value)} sx={{ mt: 2 }} />
        <Button component="label" variant="outlined" sx={{ mt: 2 }}>
          Aggiungi immagini al dataset
          <input hidden multiple type="file" accept="image/*" onChange={add} />
        </Button>
        <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
          {images.length} immagini nel dataset
        </Typography>
        <TextField type="number" label="Step" value={steps}
          onChange={(e) => setSteps(Number(e.target.value) || 2000)} sx={{ mt: 2 }} />
        <Button fullWidth disabled={busy || staging || !images.length} onClick={train} sx={{ mt: 2 }}>
          {staging ? 'Carico…' : busy ? 'Alleno…' : 'Allena'}
        </Button>
        <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 1 }}>
          Il training è lungo: puoi chiudere l'app, il server continua. Ricontrolla lo stato più tardi.
        </Typography>
        <JobPanel job={job} busy={busy} error={error} />
      </Box>
    </FeatureGate>
  );
}
