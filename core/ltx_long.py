"""Soglia della passata singola per i video LTX LUNGHI, per release + risoluzione + fps (punto 29, 31/10/2026).

Fonti (docs ufficiali ltx-2-5 e ltx-pipelines, verificate da Opus il 31/10):
  · LTX-2.5: max 20 s a 720p/1080p con 24/25 fps; 10 s a 48/50 fps o a 1440p/4K.
  · LTX-2.3: 20 s (invariato).
  · `seconds_to_clamped_num_frames` ha max_frames=1024 (~42 s a 24 fps): è una guardia anti-OOM, non un limite di addestramento.
Oltre la soglia il video si spezza in blocchi (ltx2_infer._run_chunked). Con la 2.5 e prompt per inquadratura (`prompts`)
i blocchi sono INQUADRATURE separate: nessun frame-guida dal blocco precedente e unione con stacco netto.
"""
from __future__ import annotations


def single_pass_max_seconds(release: str | None, width: int, height: int, fps: float) -> float:
    rel = str(release or "2.3")
    if rel.startswith("2.5"):
        high_res = max(int(width or 0), int(height or 0)) > 1920 or min(int(width or 0), int(height or 0)) > 1080
        high_fps = float(fps or 24) > 25.5
        return 10.0 if (high_res or high_fps) else 20.0
    return 20.0


def single_pass_max_frames(release, width, height, fps, user_threshold_s: float | None = None) -> int:
    """Frame 8n+1 della passata singola. `user_threshold_s` = soglia scelta in Avanzate → Video lunghi (tetto, mai oltre il limite della release)."""
    s = single_pass_max_seconds(release, width, height, fps)
    if user_threshold_s:
        s = min(s, float(user_threshold_s))
    return int(s * float(fps)) // 8 * 8 + 1
