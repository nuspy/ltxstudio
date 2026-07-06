import { useState } from 'react';
import { Box, Button, TextField, Typography } from '@mui/material';
import { FeatureGate } from '../components/FeatureGate';
import { ModelPicker, toArch, type Components } from '../components/ModelPicker';
import { JobPanel } from '../components/JobPanel';
import { useJob } from '../hooks/useJob';

export default function Images() {
  const [model, setModel] = useState('');
  const [comp, setComp] = useState<Components>({});
  const [prompt, setPrompt] = useState('');
  const { job, busy, error, run } = useJob();

  const gen = () =>
    run('/image/generate', {
      prompt,
      model,
      arch: toArch(comp.arch),
      steps: comp.steps ?? 20,
      guidance: comp.cfg ?? 3.5,
    });

  return (
    <FeatureGate cap="cap.gen_images">
      <Box sx={{ p: 2 }}>
        <Typography variant="h6" sx={{ mb: 2 }}>Genera immagine</Typography>
        <ModelPicker task="image" value={model} onChange={setModel} onComponents={setComp} />
        <TextField
          label="Descrizione (prompt)"
          fullWidth
          multiline
          minRows={2}
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          sx={{ my: 2 }}
        />
        <Button fullWidth disabled={busy || !model || !prompt} onClick={gen}>
          {busy ? 'Genero…' : 'Genera'}
        </Button>
        <JobPanel job={job} busy={busy} error={error} />
      </Box>
    </FeatureGate>
  );
}
