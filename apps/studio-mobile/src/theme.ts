import { createTheme } from '@mui/material/styles';

// Tema scuro "pro-max", coerente con la palette desktop (#101014 bg, accent #74c0fc).
export const darkTheme = createTheme({
  palette: {
    mode: 'dark',
    primary: { main: '#74c0fc' },
    secondary: { main: '#b197fc' },
    success: { main: '#2f9e44' },
    error: { main: '#ff6b6b' },
    background: { default: '#101014', paper: '#1a1a1f' },
  },
  shape: { borderRadius: 12 },
  typography: { fontFamily: 'Inter, Roboto, system-ui, sans-serif' },
  components: {
    MuiButton: { defaultProps: { variant: 'contained' }, styleOverrides: { root: { textTransform: 'none', fontWeight: 600 } } },
    MuiCard: { styleOverrides: { root: { border: '1px solid #2a2a32' } } },
  },
});
