import { Box, Typography } from '@mui/material';

// Placeholder temporaneo per le schermate riempite nelle fasi S2–S5.
export default function Stub({ title, phase }: { title: string; phase: string }) {
  return (
    <Box sx={{ p: 3, textAlign: 'center', color: 'text.secondary' }}>
      <Typography variant="h6" gutterBottom>{title}</Typography>
      <Typography variant="body2">In costruzione — {phase}.</Typography>
    </Box>
  );
}
