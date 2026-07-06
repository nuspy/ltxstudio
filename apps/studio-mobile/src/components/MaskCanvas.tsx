import { forwardRef, useEffect, useImperativeHandle, useRef, useState } from 'react';
import { Box, Slider, Stack, Typography } from '@mui/material';

export type MaskCanvasHandle = {
  getImageBlob: () => Promise<Blob | null>;
  getMaskBlob: () => Promise<Blob | null>;
  clear: () => void;
};

// Editor mask su HTML5 Canvas: due layer (immagine + mask trasparente); pennello/dimensione.
// La mask è bianca dove dipinta su nero → pronta per /image/inpaint.
export const MaskCanvas = forwardRef<MaskCanvasHandle, { src: string }>(({ src }, ref) => {
  const imgRef = useRef<HTMLCanvasElement>(null);
  const maskRef = useRef<HTMLCanvasElement>(null);
  const drawing = useRef(false);
  const [brush, setBrush] = useState(30);
  const dims = useRef({ w: 512, h: 512 });

  useEffect(() => {
    const im = new Image();
    im.onload = () => {
      const scale = Math.min(1, 1024 / im.width);
      const w = Math.round(im.width * scale);
      const h = Math.round(im.height * scale);
      dims.current = { w, h };
      const ic = imgRef.current!;
      const mc = maskRef.current!;
      ic.width = mc.width = w;
      ic.height = mc.height = h;
      ic.getContext('2d')!.drawImage(im, 0, 0, w, h);
      const mx = mc.getContext('2d')!;
      mx.fillStyle = 'black';
      mx.fillRect(0, 0, w, h);
    };
    im.src = src;
  }, [src]);

  const at = (e: React.PointerEvent) => {
    const c = maskRef.current!;
    const r = c.getBoundingClientRect();
    return { x: (e.clientX - r.left) * (c.width / r.width), y: (e.clientY - r.top) * (c.height / r.height) };
  };
  const paint = (e: React.PointerEvent) => {
    if (!drawing.current) return;
    const { x, y } = at(e);
    const mx = maskRef.current!.getContext('2d')!;
    mx.fillStyle = 'white';
    mx.beginPath();
    mx.arc(x, y, brush, 0, Math.PI * 2);
    mx.fill();
  };

  useImperativeHandle(ref, () => ({
    getImageBlob: () => new Promise((res) => imgRef.current!.toBlob(res, 'image/png')),
    getMaskBlob: () => new Promise((res) => maskRef.current!.toBlob(res, 'image/png')),
    clear: () => {
      const mx = maskRef.current!.getContext('2d')!;
      mx.fillStyle = 'black';
      mx.fillRect(0, 0, dims.current.w, dims.current.h);
    },
  }));

  return (
    <Box>
      <Box sx={{ position: 'relative', width: '100%' }}>
        <canvas ref={imgRef} style={{ width: '100%', borderRadius: 8, display: 'block' }} />
        <canvas
          ref={maskRef}
          style={{ position: 'absolute', top: 0, left: 0, width: '100%', opacity: 0.5, touchAction: 'none', cursor: 'crosshair' }}
          onPointerDown={(e) => { drawing.current = true; paint(e); }}
          onPointerMove={paint}
          onPointerUp={() => { drawing.current = false; }}
          onPointerLeave={() => { drawing.current = false; }}
        />
      </Box>
      <Stack direction="row" spacing={2} alignItems="center" sx={{ mt: 1 }}>
        <Typography variant="body2">Pennello</Typography>
        <Slider min={5} max={100} value={brush} onChange={(_, v) => setBrush(v as number)} sx={{ flex: 1 }} />
      </Stack>
    </Box>
  );
});
