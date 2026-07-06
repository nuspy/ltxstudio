import { useEffect, useState } from 'react';
import { Autocomplete, Box, Chip, TextField } from '@mui/material';
import { apiGet } from '../api';

export type Components = {
  arch?: string;
  vae?: string;
  text_encoders?: string[];
  steps?: number;
  cfg?: number;
  loras_compatible?: string[];
};

// family (dal detector) → arch atteso dagli endpoint immagine
const ARCH_MAP: Record<string, string> = { flux1: 'flux', flux2: 'flux2', zimage: 'zimage', ernie: 'ernie' };
export function toArch(family?: string): string {
  if (!family) return 'flux';
  return ARCH_MAP[family] || family;
}

// Selettore modello: scelto il modello, mostra i COMPONENTI AUTO (read-only) e li propaga al chiamante.
export function ModelPicker({
  task, value, onChange, onComponents,
}: {
  task: string;
  value: string;
  onChange: (m: string) => void;
  onComponents: (c: Components) => void;
}) {
  const [models, setModels] = useState<string[]>([]);
  const [comp, setComp] = useState<Components>({});

  useEffect(() => {
    apiGet(`/models?task=${task}`)
      .then((d) => {
        setModels(d.models || []);
        if (!value && d.recommended?.model) onChange(d.recommended.model);
      })
      .catch(() => {});
  }, [task]);

  useEffect(() => {
    if (!value) return;
    apiGet(`/components?model=${encodeURIComponent(value)}&task=${task}`)
      .then((c: Components) => {
        setComp(c);
        onComponents(c);
      })
      .catch(() => {});
  }, [value]);

  return (
    <Box>
      <Autocomplete
        options={models}
        value={value || null}
        onChange={(_, v) => onChange(v || '')}
        renderInput={(p) => <TextField {...p} label="Modello" />}
      />
      <Box sx={{ mt: 1, display: 'flex', gap: 0.5, flexWrap: 'wrap' }}>
        {comp.arch && <Chip size="small" color="primary" variant="outlined" label={`arch: ${comp.arch}`} />}
        {comp.vae && <Chip size="small" variant="outlined" label={`VAE: ${comp.vae}`} />}
        {(comp.text_encoders || []).map((t) => <Chip key={t} size="small" variant="outlined" label={t} />)}
        {comp.steps != null && <Chip size="small" variant="outlined" label={`${comp.steps} step`} />}
      </Box>
    </Box>
  );
}
