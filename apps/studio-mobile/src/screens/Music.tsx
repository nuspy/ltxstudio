import { useState } from 'react';
import { Box, Button, Checkbox, FormControlLabel, MenuItem, TextField, Typography } from '@mui/material';
import { FeatureGate } from '../components/FeatureGate';
import { JobPanel } from '../components/JobPanel';
import { useJob } from '../hooks/useJob';
import { isEnabled } from '../features';

type Mode = 'song' | 'videoclip';

export default function Music() {
  const [mode, setMode] = useState<Mode>('song');
  const [name, setName] = useState('');
  const [tags, setTags] = useState('');
  const [lyrics, setLyrics] = useState('');
  const [dur, setDur] = useState(30);
  const [instr, setInstr] = useState(false);
  const { job, busy, error, run } = useJob();

  const gen = () =>
    mode === 'song'
      ? run('/music/song', { name: name || 'song', tags, lyrics: instr ? '' : lyrics, duration_s: dur, instrumental: instr })
      : run('/music/music_video', { lyrics, scene_seconds: 4 });

  const canGen = !busy && (mode === 'song' ? (instr || !!lyrics || !!tags) : !!lyrics);

  return (
    <FeatureGate cap="cap.music_song">
      <Box sx={{ p: 2 }}>
        <Typography variant="h6" sx={{ mb: 2 }}>Musica</Typography>
        <TextField select fullWidth label="Cosa creare" value={mode} onChange={(e) => setMode(e.target.value as Mode)}>
          <MenuItem value="song">Canzone / strumentale</MenuItem>
          {isEnabled('cap.music_video') && <MenuItem value="videoclip">Videoclip (da testo)</MenuItem>}
        </TextField>

        {mode === 'song' && (
          <>
            <TextField label="Titolo" fullWidth value={name} onChange={(e) => setName(e.target.value)} sx={{ mt: 2 }} />
            <TextField label="Stile / tag (es. pop, chitarra, energico)" fullWidth value={tags}
              onChange={(e) => setTags(e.target.value)} sx={{ mt: 2 }} />
            <FormControlLabel
              control={<Checkbox checked={instr} onChange={(e) => setInstr(e.target.checked)} />}
              label="Strumentale (senza voce)"
            />
          </>
        )}

        {!(mode === 'song' && instr) && (
          <TextField label="Testo" fullWidth multiline minRows={3} value={lyrics}
            onChange={(e) => setLyrics(e.target.value)} sx={{ mt: 2 }} />
        )}
        {mode === 'song' && (
          <TextField type="number" label="Durata (s)" value={dur}
            onChange={(e) => setDur(Number(e.target.value) || 30)} sx={{ mt: 2 }} />
        )}

        <Button fullWidth disabled={!canGen} onClick={gen} sx={{ mt: 2 }}>
          {busy ? 'Genero…' : 'Genera'}
        </Button>
        <JobPanel job={job} busy={busy} error={error} />
      </Box>
    </FeatureGate>
  );
}
