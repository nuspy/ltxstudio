import { ReactNode } from 'react';
import { Box, Button, Card, CardContent, Typography } from '@mui/material';
import LockIcon from '@mui/icons-material/Lock';
import { isEnabled, upsell } from '../features';

// Rispetta il feature matrix: se la capability è disabilitata nell'edizione, mostra un pannello upsell
// (stessa filosofia 🔒 del desktop) al posto del contenuto — dentro la stessa regione.
export function FeatureGate({ cap, children }: { cap: string; children: ReactNode }) {
  if (isEnabled(cap)) return <>{children}</>;
  const u = upsell(cap);
  return (
    <Box sx={{ p: 3, display: 'flex', justifyContent: 'center' }}>
      <Card sx={{ maxWidth: 420, textAlign: 'center' }}>
        <CardContent>
          <LockIcon sx={{ fontSize: 40, color: 'primary.main', mb: 1 }} />
          <Typography variant="h6" gutterBottom>
            Funzione non inclusa
          </Typography>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
            {u?.tier_name
              ? `Disponibile con il piano ${u.tier_name}${u.price ? ` (${u.price})` : ''}.`
              : 'Disponibile in un piano superiore.'}
          </Typography>
          {u?.url && (
            <Button onClick={() => window.open(u.url!, '_blank')}>Sblocca</Button>
          )}
        </CardContent>
      </Card>
    </Box>
  );
}
