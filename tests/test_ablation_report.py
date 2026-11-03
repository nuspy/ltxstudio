import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "app_live"))
import ablation_report as AR  # noqa: E402


def test_table_and_comparison_image(tmp_path):
    rows = [{"id": "abl_A0_base", "k_mediana": -12.0, "quota_fantasma": 0.6, "fedelta_testa": 0.007, "fuori_sagoma": 0.2, "volto_fantasma": 0.5, "oldbg": "10.0 %", "video": ""},
            {"id": "abl_A1", "k_mediana": None, "quota_fantasma": None, "oldbg": "n/d", "video": ""}]
    t = AR.table(rows)
    assert "abl_A0_base" in t and "-12" in t and "n/d" in t and t.count("\n") >= 4
    AR.comparison_image(rows, str(tmp_path / "c.jpg"))                       # video assente: tasselli neri, nessuna eccezione
    assert (tmp_path / "c.jpg").exists()


def test_winpath():
    assert AR.winpath("/e/ai/x/y.mp4") == "E:/ai/x/y.mp4" and AR.winpath("C:/a/b") == "C:/a/b"
