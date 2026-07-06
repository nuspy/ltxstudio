import { Alert, Box, Button, Card, CardContent, LinearProgress, Typography } from '@mui/material';
import DownloadIcon from '@mui/icons-material/Download';
import type { Job, JobFile } from '../api';
import { saveToDevice } from '../download';

function isVideo(name: string) {
  return /\.(mp4|webm|mkv|mov)$/i.test(name);
}

function ResultFile({ f }: { f: JobFile }) {
  return (
    <Box sx={{ mt: 1 }}>
      {isVideo(f.name) ? (
        <video src={f.download_url} controls style={{ width: '100%', borderRadius: 8 }} />
      ) : (
        <img src={f.download_url} alt={f.name} style={{ width: '100%', borderRadius: 8 }} />
      )}
      <Button
        fullWidth
        startIcon={<DownloadIcon />}
        sx={{ mt: 1 }}
        onClick={() => saveToDevice(f.download_url, f.name)}
      >
        Scarica
      </Button>
    </Box>
  );
}

export function JobPanel({ job, busy, error }: { job: Job | null; busy: boolean; error: string }) {
  if (error) return <Alert severity="error" sx={{ mt: 2 }}>{error}</Alert>;
  if (!job && !busy) return null;
  const pct = Math.round((job?.progress ?? 0) * 100);
  const done = job?.status === 'success';
  return (
    <Card sx={{ mt: 2 }}>
      <CardContent>
        {!done && (
          <>
            <Typography variant="body2" color="text.secondary">
              {job?.message || 'in coda…'}
              {job?.queue_position ? ` (coda #${job.queue_position})` : ''}
            </Typography>
            <LinearProgress
              variant={pct > 1 ? 'determinate' : 'indeterminate'}
              value={pct}
              sx={{ my: 1 }}
            />
          </>
        )}
        {done && job?.files?.map((f) => <ResultFile key={f.path} f={f} />)}
      </CardContent>
    </Card>
  );
}
