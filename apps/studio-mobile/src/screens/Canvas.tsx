import { useRef, useState } from 'react';
import { Box, Button, TextField, Typography } from '@mui/material';
import { FeatureGate } from '../components/FeatureGate';
import { ModelPicker, toArch, type Components } from '../components/ModelPicker';
import { MaskCanvas, type MaskCanvasHandle } from '../components/MaskCanvas';
import { JobPanel } from '../components/JobPanel';
import { useJob } from '../hooks/useJob';
import { uploadFile } from '../api';

export default function Canvas() {
  const [src, setSrc] = useState('');
  const [model, setModel] = useState('');
  const [comp, setComp] = useState<Components>({});
  const [prompt, setPrompt] = useState('');
  const [staging, setStaging] = useState(false);
  const mc = useRef<MaskCanvasHandle>(null);
  const { job, busy, error, run } = useJob();

  const pick = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (!f) return;
    const r = new FileReader();
    r.onload = () => setSrc(r.result as string);
    r.readAsDataURL(f);
  };

  const gen = async () => {
    if (!mc.current) return;
    setStaging(true);
    try {
      const ib = await mc.current.getImageBlob();
      const mb = await mc.current.getMaskBlob();
      if (!ib || !mb) return;
      const up1 = await uploadFile(ib, 'canvas.png');
      const up2 = await uploadFile(mb, 'mask.png');
      await run('/image/inpaint', {
        prompt,
        model,
        arch: toArch(comp.arch) === 'flux2' ? 'flux2_inpaint' : 'inpaint',
        image_path: up1.path,
        mask_path: up2.path,
        steps: comp.steps ?? 20,
      });
    } catch (e) {
      /* gestito da JobPanel via error nel run; upload error qui */
    } finally {
      setStaging(false);
    }
  };

  return (
    <FeatureGate cap="cap.inpaint">
      <Box sx={{ p: 2 }}>
        <Typography variant="h6" sx={{ mb: 2 }}>Canvas &amp; Mask (inpaint)</Typography>
        <Button component="label" variant="outlined">
          Scegli immagine
          <input hidden type="file" accept="image/*" onChange={pick} />
        </Button>
        {src && (
          <Box sx={{ mt: 2 }}>
            <MaskCanvas ref={mc} src={src} />
            <Box sx={{ mt: 1 }}>
              <Button size="small" onClick={() => mc.current?.clear()}>Pulisci mask</Button>
            </Box>
            <Box sx={{ mt: 2 }}>
              <ModelPicker task="image" value={model} onChange={setModel} onComponents={setComp} />
            </Box>
            <TextField
              label="Cosa disegnare nell'area mascherata"
              fullWidth
              multiline
              minRows={2}
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              sx={{ my: 2 }}
            />
            <Button fullWidth disabled={busy || staging || !model || !prompt} onClick={gen}>
              {staging ? 'Carico…' : busy ? 'Genero…' : 'Genera inpaint'}
            </Button>
            <JobPanel job={job} busy={busy} error={error} />
          </Box>
        )}
      </Box>
    </FeatureGate>
  );
}
