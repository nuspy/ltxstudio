from core import ltx_long as L


def test_soglie_per_release_risoluzione_fps():
    assert L.single_pass_max_seconds("2.3", 1280, 704, 24) == 20.0
    assert L.single_pass_max_seconds("2.3", 3840, 2160, 50) == 20.0           # 2.3 invariata
    assert L.single_pass_max_seconds("2.5", 1920, 1080, 24) == 20.0
    assert L.single_pass_max_seconds("2.5", 1280, 720, 25) == 20.0
    assert L.single_pass_max_seconds("2.5", 1280, 720, 48) == 10.0
    assert L.single_pass_max_seconds("2.5", 2560, 1440, 24) == 10.0
    assert L.single_pass_max_seconds("2.5", 3840, 2160, 24) == 10.0


def test_frame_8n_piu_1_e_tetto_utente():
    assert L.single_pass_max_frames("2.3", 704, 1280, 24) == 481
    assert L.single_pass_max_frames("2.5", 704, 1280, 24, user_threshold_s=12) == 12 * 24 // 8 * 8 + 1
    assert L.single_pass_max_frames("2.5", 704, 1280, 24, user_threshold_s=30) == 481      # mai oltre la release
