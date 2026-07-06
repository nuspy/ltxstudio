import { useNavigate } from 'react-router-dom';
import { Box, Card, CardActionArea, CardContent, Chip, Grid, Typography } from '@mui/material';
import ImageIcon from '@mui/icons-material/Image';
import BrushIcon from '@mui/icons-material/Brush';
import MovieIcon from '@mui/icons-material/Movie';
import TuneIcon from '@mui/icons-material/Tune';
import SchoolIcon from '@mui/icons-material/School';
import MusicNoteIcon from '@mui/icons-material/MusicNote';
import LockIcon from '@mui/icons-material/Lock';
import { isEnabled } from '../features';

type Fn = { path: string; label: string; cap: string; icon: JSX.Element };

const FUNCTIONS: Fn[] = [
  { path: '/images', label: 'Immagini', cap: 'cap.gen_images', icon: <ImageIcon /> },
  { path: '/canvas', label: 'Canvas & Mask', cap: 'cap.inpaint', icon: <BrushIcon /> },
  { path: '/video', label: 'Video + ControlNet', cap: 'cap.video', icon: <MovieIcon /> },
  { path: '/video?mode=controlnet', label: 'ControlNet', cap: 'cap.controlnet', icon: <TuneIcon /> },
  { path: '/training', label: 'Training LoRA', cap: 'cap.lora_image_train', icon: <SchoolIcon /> },
  { path: '/music', label: 'Musica & Videoclip', cap: 'cap.music_song', icon: <MusicNoteIcon /> },
];

export default function Home() {
  const nav = useNavigate();
  return (
    <Box sx={{ p: 2 }}>
      <Typography variant="h6" sx={{ mb: 2 }}>Cosa vuoi creare?</Typography>
      <Grid container spacing={2}>
        {FUNCTIONS.map((f) => {
          const on = isEnabled(f.cap);
          return (
            <Grid item xs={6} key={f.path + f.label}>
              <Card sx={{ opacity: on ? 1 : 0.6 }}>
                <CardActionArea onClick={() => nav(f.path)} sx={{ minHeight: 110 }}>
                  <CardContent sx={{ textAlign: 'center' }}>
                    <Box sx={{ color: 'primary.main', mb: 1 }}>{f.icon}</Box>
                    <Typography variant="subtitle2">{f.label}</Typography>
                    {!on && <Chip size="small" icon={<LockIcon />} label="Bloccato" sx={{ mt: 1 }} />}
                  </CardContent>
                </CardActionArea>
              </Card>
            </Grid>
          );
        })}
      </Grid>
    </Box>
  );
}
