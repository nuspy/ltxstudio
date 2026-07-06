import { useEffect, useState } from 'react';
import { Routes, Route, useNavigate, useLocation } from 'react-router-dom';
import { AppBar, Box, BottomNavigation, BottomNavigationAction, Paper, Toolbar, Typography } from '@mui/material';
import HomeIcon from '@mui/icons-material/Home';
import CollectionsIcon from '@mui/icons-material/Collections';
import VerifiedIcon from '@mui/icons-material/Verified';
import SettingsIcon from '@mui/icons-material/Settings';
import { initApi, fetchFeatures } from './api';
import { setServerSnapshot, edition } from './features';
import Home from './screens/Home';
import Settings from './screens/Settings';
import Detector from './screens/Detector';
import Gallery from './screens/Gallery';
import Images from './screens/Images';
import Canvas from './screens/Canvas';
import Video from './screens/Video';
import Training from './screens/Training';
import Music from './screens/Music';

const NAV = [
  { path: '/', label: 'Home', icon: <HomeIcon /> },
  { path: '/gallery', label: 'Galleria', icon: <CollectionsIcon /> },
  { path: '/detector', label: 'Rilevatore', icon: <VerifiedIcon /> },
  { path: '/settings', label: 'Impostazioni', icon: <SettingsIcon /> },
];

export default function App() {
  const nav = useNavigate();
  const loc = useLocation();
  const [ready, setReady] = useState(false);

  useEffect(() => {
    (async () => {
      await initApi();
      try {
        const snap = await fetchFeatures();
        setServerSnapshot(snap);
      } catch {
        // server offline: si resta sul manifest bakeato (o fail-open in dev)
      }
      setReady(true);
    })();
  }, []);

  const navIndex = Math.max(0, NAV.findIndex((n) => n.path === loc.pathname));

  return (
    <Box sx={{ pb: 7, minHeight: '100vh' }}>
      <AppBar position="sticky" color="default" elevation={0} sx={{ borderBottom: '1px solid #2a2a32' }}>
        <Toolbar variant="dense">
          <Typography variant="h6" sx={{ flexGrow: 1, color: 'primary.main', fontWeight: 700 }}>
            LTX Studio
          </Typography>
          <Typography variant="caption" color="text.secondary">
            {edition()}
          </Typography>
        </Toolbar>
      </AppBar>

      <Box sx={{ maxWidth: 720, mx: 'auto' }}>
        {ready ? (
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/images" element={<Images />} />
            <Route path="/canvas" element={<Canvas />} />
            <Route path="/video" element={<Video />} />
            <Route path="/training" element={<Training />} />
            <Route path="/music" element={<Music />} />
            <Route path="/detector" element={<Detector />} />
            <Route path="/gallery" element={<Gallery />} />
            <Route path="/settings" element={<Settings />} />
          </Routes>
        ) : (
          <Box sx={{ p: 4, textAlign: 'center', color: 'text.secondary' }}>Avvio…</Box>
        )}
      </Box>

      <Paper sx={{ position: 'fixed', bottom: 0, left: 0, right: 0 }} elevation={3}>
        <BottomNavigation
          showLabels
          value={navIndex}
          onChange={(_, i) => nav(NAV[i].path)}
        >
          {NAV.map((n) => (
            <BottomNavigationAction key={n.path} label={n.label} icon={n.icon} />
          ))}
        </BottomNavigation>
      </Paper>
    </Box>
  );
}
