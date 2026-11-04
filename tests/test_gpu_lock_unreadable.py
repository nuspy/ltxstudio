"""Lock GPU lasciato a zeri da un riavvio del PC (04/11, turni 43r e 44): 72 byte NUL in data/gpu.lock, holder() = None, il ramo «proprietario morto» non scattava mai
e OGNI job della GUI restava in coda per ore senza una riga di log. Ora un lock illeggibile e vecchio e orfano; uno appena creato resta rispettato; l'attesa si scrive nel log."""
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core import gpu_lock  # noqa: E402


def _lock_of_zeros(tmp_path, monkeypatch, age_s):
    path = tmp_path / "gpu.lock"
    monkeypatch.setattr(gpu_lock, "_LOCK_PATH", str(path))
    gpu_lock._STATE["count"] = 0
    path.write_bytes(b"\0" * 72)
    old = time.time() - age_s
    os.utime(path, (old, old))
    return path


def test_old_unreadable_lock_is_removed_and_taken_with_a_log_line(tmp_path, monkeypatch, capsys):
    _lock_of_zeros(tmp_path, monkeypatch, age_s=3600)
    assert gpu_lock.holder() is None
    assert gpu_lock.acquire("t:dopo-riavvio", timeout_s=5)
    assert "lock orfano illeggibile" in capsys.readouterr().out
    assert gpu_lock.holder()["tag"] == "t:dopo-riavvio"
    gpu_lock.release()


def test_fresh_unreadable_lock_is_respected_because_its_creator_may_still_be_writing(tmp_path, monkeypatch):
    _lock_of_zeros(tmp_path, monkeypatch, age_s=1)
    assert gpu_lock.acquire("t:subito", timeout_s=0) is False


def test_lock_file_is_written_and_synced_with_its_owner(tmp_path, monkeypatch):
    monkeypatch.setattr(gpu_lock, "_LOCK_PATH", str(tmp_path / "gpu.lock"))
    gpu_lock._STATE["count"] = 0
    assert gpu_lock.acquire("t:scrittura", timeout_s=1)
    assert gpu_lock.holder()["pid"] == os.getpid()
    gpu_lock.release()


def test_queue_logs_who_holds_the_gpu_every_minute_while_a_job_waits(capsys):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from ui.job_queue import GenerationQueue
    queue = GenerationQueue()

    class Job:
        label = "Video edit"
    queue._log_gpu_wait(Job(), {"pid": 4242, "tag": "training"})
    queue._log_gpu_wait(Job(), {"pid": 4242, "tag": "training"})            # entro un minuto: nessuna seconda riga
    queue._log_gpu_wait(Job(), {})
    lines = [line for line in capsys.readouterr().out.splitlines() if line.startswith("[gpu-lock]")]
    assert len(lines) == 1 and "pid 4242 (training)" in lines[0] and "«Video edit»" in lines[0]
    queue._gpu_wait_logged -= 61
    queue._log_gpu_wait(Job(), {})
    assert "illeggibile" in capsys.readouterr().out
