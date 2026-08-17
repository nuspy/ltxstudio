"""Regressione: scelta della maschera SAM2 quando i box sono più di uno.

`post_process_masks(...)[0]` toglie la dimensione di BATCH alle maschere ma NON agli `iou_scores`,
che restano `(1, n_box, n_mask)`. Con un solo box non si vedeva; con due box `scores[b]` era ancora
2-D e `argmax()` (sul tensore appiattito) restituiva un indice fino a `n_box*n_mask-1`, fuori dai 3
canali multimask → `IndexError: index 5 is out of bounds for dimension 1 with size 3`, e la
segmentazione a parole (quindi anche la rimozione oggetti, che la usa per costruire la maschera)
falliva su qualunque frase che pescasse più di un oggetto.

Niente GPU e niente modelli: solo la forma dei tensori.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inference_server.backends.image_segment import _mask_to_2d  # noqa: E402

H = W = 8


def _masks(n_box: int, n_mask: int = 3) -> torch.Tensor:
    """Maschere finte distinguibili: il canale k del box b copre le prime (k+1) righe."""
    m = torch.zeros(n_box, n_mask, H, W)
    for b in range(n_box):
        for k in range(n_mask):
            m[b, k, : k + 1, :] = 1.0
    return m


def test_due_box_con_scores_batched_non_solleva():
    masks = _masks(2)
    scores = torch.tensor([[[0.1, 0.9, 0.2], [0.8, 0.1, 0.1]]])      # (1, 2, 3) ← con batch
    out = _mask_to_2d(masks, scores)
    assert out.shape == (H, W)
    # box 0 → canale 1 (2 righe), box 1 → canale 0 (1 riga); l'unione ne copre 2
    assert out[:2].max() == 255 and out[2:].max() == 0


def test_un_box_senza_batch_sceglie_il_punteggio_migliore():
    masks = _masks(1)
    scores = torch.tensor([[0.1, 0.2, 0.9]])                          # (1, 3) ← senza batch
    out = _mask_to_2d(masks, scores)
    assert out[:3].max() == 255 and out[3:].max() == 0                # canale 2 = 3 righe


def test_scores_di_forma_inattesa_ripiega_senza_sollevare():
    masks = _masks(2)
    out = _mask_to_2d(masks, torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5]))  # forma che non combacia
    assert out.shape == (H, W)
    assert out[:1].max() == 255                                        # ripiego sulla prima maschera


def test_senza_scores_resta_l_unione():
    out = _mask_to_2d(_masks(1), None)
    assert out.shape == (H, W)
    assert out[:3].max() == 255
