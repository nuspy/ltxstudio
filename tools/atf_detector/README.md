# ATF Detector — app standalone

Verifica se una **foto o un video** è stato generato con **AI Tools Forge**, leggendo la firma di
provenienza invisibile (#107) e l'eventuale **testo custom**. Funziona su qualsiasi fonte: file originale,
**screenshot**, foto dello schermo (anche leggermente storta). Nessun originale necessario.

## Avvio (dal repo)
```
python tools/atf_detector/detector_gui.py
```
Trascina i file nella finestra, oppure "Apri file…" / "Scansiona cartella…".

## Compilazione in .exe autonomo (distribuibile)
```
python tools/atf_detector/build_detector.py
```
Produce `tools/atf_detector/dist/ATF-Detector.exe` — bundle **indipendente** che contiene solo la logica di
rilevamento (`core/stego.py`, con la chiave) + numpy + opencv + PySide6. NON include il resto di LTX Studio.

## Versione a riga di comando
`python scripts/atf_detect.py <file>` (o `--dir <cartella>`, `--json`). Exit-code 0 = firma trovata.

## Note
- La logica vive in **`core/stego.py`** (unica fonte). L'app la importa → nessuna duplicazione/derapata.
- Sicurezza: la chiave è nel binario compilato. Per offuscazione più forte usare Nuitka; per garanzia
  massima, un Detector lato-server che non espone la chiave.
- Limite noto: foto dello schermo con **prospettiva forte/keystone** non ancora coperta (resync = fase 2b);
  il **testo** custom è di ~5 caratteri (limite fisico) e si perde sotto rotazione forte (la firma sì/no no).
