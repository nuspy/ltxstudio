"""Analizzatore impostazioni LTX Studio.

Controlla la configurazione corrente e segnala problemi (errori bloccanti, avvisi, info).
Ogni regola ritorna una lista di AnalyzerIssue.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


Severity = Literal["error", "warning", "info"]


@dataclass
class AnalyzerIssue:
    severity: Severity
    code: str                    # identificatore unico della regola
    message: str                 # messaggio leggibile
    field: str | None = None     # campo GUI interessato (es. "cfg", "steps")
    autofix: dict | None = None  # se non None → dict con i parametri corretti da applicare


@dataclass
class AnalysisResult:
    issues: list[AnalyzerIssue] = field(default_factory=list)

    @property
    def errors(self) -> list[AnalyzerIssue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[AnalyzerIssue]:
        return [i for i in self.issues if i.severity == "warning"]

    @property
    def infos(self) -> list[AnalyzerIssue]:
        return [i for i in self.issues if i.severity == "info"]

    @property
    def has_errors(self) -> bool:
        return bool(self.errors)


class SettingsAnalyzer:
    """Valida la configurazione corrente e ritorna un AnalysisResult."""

    # Modelli noti come "distilled" (guidanza CFG bassa, tipicamente < 1.5)
    _DISTILLED_KEYWORDS = ("distilled", "gguf", "q4_k", "q5_k", "q8_0", "ltx-2.3-22b-d", "22b_distilled")

    def analyze(self,
                steps: int,
                cfg: float,
                width: int,
                height: int,
                nsfw_mode: str,
                model_name: str,
                loras: list[dict],
                sampler: str = "",
                scheduler: str = "",
                audio_mode: str = "native",
                ) -> AnalysisResult:
        result = AnalysisResult()
        mn = model_name.lower()
        is_distilled = any(kw in mn for kw in self._DISTILLED_KEYWORDS)

        # ── Steps ──────────────────────────────────────────────────────────────
        if steps < 4:
            result.issues.append(AnalyzerIssue(
                severity="error", code="steps_too_low", field="steps",
                message=f"Steps={steps} è troppo basso: l'output sarà degradato o vuoto. Minimo consigliato: 4.",
                autofix={"steps": 4},
            ))
        elif steps < 8:
            result.issues.append(AnalyzerIssue(
                severity="warning", code="steps_low", field="steps",
                message=f"Steps={steps} può produrre output di qualità ridotta. Consigliato ≥ 8 per test, ≥ 20 per qualità.",
            ))
        elif steps > 60:
            result.issues.append(AnalyzerIssue(
                severity="info", code="steps_high", field="steps",
                message=f"Steps={steps} è molto alto: raramente porta benefici oltre 50 per LTX-2.3. "
                        "Considera 30-40 per il miglior rapporto qualità/tempo.",
            ))

        # ── CFG ────────────────────────────────────────────────────────────────
        # NB: LTX Studio inietta sempre NAG + sampler `euler_ancestral_cfg_pp` (vedi quality
        # upgrade). In questo regime la guida negativa la fornisce il NAG e il CFGGuider resta
        # a ~1.0 per TUTTI i modelli (distilled e base/dev). Quindi CFG 1.0 è corretto ovunque;
        # avvisiamo solo se è chiaramente fuori scala.
        if is_distilled and cfg > 2.5:
            result.issues.append(AnalyzerIssue(
                severity="error", code="cfg_too_high_distilled", field="cfg",
                message=f"CFG={cfg:.2f} è troppo alta per un modello distilled ('{model_name}'). "
                        "Con NAG + sampler cfg_pp i distilled vanno a CFG ~1.0 (max ~1.5).",
                autofix={"cfg": 1.0},
            ))
        elif cfg > 7.0:
            result.issues.append(AnalyzerIssue(
                severity="warning", code="cfg_very_high", field="cfg",
                message=f"CFG={cfg:.2f} è molto alta: può causare artefatti e sovra-saturazione. "
                        "Con NAG attivo, CFG 1.0 è di norma sufficiente.",
            ))
        elif cfg < 0.5:
            result.issues.append(AnalyzerIssue(
                severity="warning", code="cfg_very_low", field="cfg",
                message=f"CFG={cfg:.2f} è quasi nulla: la guida del prompt sarà debolissima. "
                        "Consigliato ~1.0 (la guida negativa la dà comunque il NAG).",
                autofix={"cfg": 1.0},
            ))

        # ── Risoluzione ────────────────────────────────────────────────────────
        if width % 32 != 0:
            result.issues.append(AnalyzerIssue(
                severity="error", code="width_not_multiple_32", field="width",
                message=f"Larghezza={width} non è multiplo di 32. Il VAE richiede dimensioni allineate a 32.",
                autofix={"width": (width // 32) * 32},
            ))
        if height % 32 != 0:
            result.issues.append(AnalyzerIssue(
                severity="error", code="height_not_multiple_32", field="height",
                message=f"Altezza={height} non è multiplo di 32. Il VAE richiede dimensioni allineate a 32.",
                autofix={"height": (height // 32) * 32},
            ))
        total_px = width * height
        if total_px > 1280 * 768:
            result.issues.append(AnalyzerIssue(
                severity="info", code="resolution_high", field=None,
                message=f"Risoluzione {width}×{height} ({total_px/1e6:.1f}MP) è grande: "
                        "aspettati tempi di generazione lunghi (VRAM elevata).",
            ))

        # ── LoRA ───────────────────────────────────────────────────────────────
        trigger_counts: dict[str, list[str]] = {}   # trigger → [lora_names]
        for lo in loras:
            name = lo.get("name", "?")
            strength = float(lo.get("strength", 1.0))
            entry = lo.get("_entry")

            # Strength fuori range
            if strength > 1.5:
                result.issues.append(AnalyzerIssue(
                    severity="warning", code="lora_strength_high", field="loras",
                    message=f"LoRA '{name}': strength={strength:.2f} è alta per LTX. "
                            "Valori > 1.5 tendono a produrre artefatti. Consigliato ≤ 1.2.",
                    autofix=None,  # auto-fix non sicuro (dipende dalla LoRA)
                ))
            if strength < 0.1:
                result.issues.append(AnalyzerIssue(
                    severity="warning", code="lora_strength_low", field="loras",
                    message=f"LoRA '{name}': strength={strength:.2f} è molto bassa — l'effetto sarà trascurabile.",
                ))

            # NSFW conflict
            if entry and getattr(entry, "nsfw", False) and nsfw_mode != "nsfw":
                result.issues.append(AnalyzerIssue(
                    severity="warning", code="lora_nsfw_conflict", field="nsfw_mode",
                    message=f"LoRA '{name}' è NSFW ma la modalità attuale è SFW. "
                            "I contenuti adulti possono essere soppressi o degradati.",
                ))

            # Arch incompatibility
            if entry:
                arch = getattr(entry, "architecture", "")
                if arch and arch.upper() not in ("LTX", "ANY", ""):
                    result.issues.append(AnalyzerIssue(
                        severity="error", code="lora_arch_mismatch", field="loras",
                        message=f"LoRA '{name}' ha architettura '{arch}' incompatibile con LTX-2.3.",
                    ))

            # Raccolta trigger per controllo duplicati
            if entry:
                for trig in getattr(entry, "triggers", []):
                    trigger_counts.setdefault(trig.lower(), []).append(name)

        # Trigger word condivise tra più LoRA (potenziale conflitto di condizionamento)
        for trig, names in trigger_counts.items():
            if len(names) > 1:
                result.issues.append(AnalyzerIssue(
                    severity="warning", code="lora_shared_trigger", field="loras",
                    message=f"Trigger word '{trig}' è usata da {len(names)} LoRA contemporaneamente: "
                            f"{', '.join(names)}. Può causare conflitti di stile.",
                ))

        # Troppe LoRA attive contemporaneamente
        if len(loras) > 4:
            result.issues.append(AnalyzerIssue(
                severity="info", code="too_many_loras", field="loras",
                message=f"{len(loras)} LoRA attive: molte LoRA combinate aumentano il rischio di artefatti "
                        "e rallentano la generazione. Considera di usarne ≤ 4.",
            ))

        return result
