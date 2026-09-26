"""Prompt di MiniMax H3 nel formato della guida ufficiale (docs/VIDEO_PROMPT_WRITING_GUIDE_{base,ref}_en.md).

Senza riferimenti (T2VA / I2VA / FL2VA):
    [istruzione di allineamento del keyframe, se c'è una foto]

    integrated_multimodal_description: [Shot 1] <descrizione> … (S1) says, <delivery>: <d>[Lingua] testo</d>

    overall_soundscape: …

    non_diegetic_music: … | N/A
Con riferimenti (Ref2VA): subject_definitions / summary / retention_analysis / detailed_description /
overall_soundscape / non_diegetic_music, con <Subject N> per le persone e <Audio N> per il timbro.

Le battute accettano i marcatori espressivi di core/voice_markup: dentro <d> va SOLO il testo (pulito), il
modo di dire la battuta ("whispers", "quickly"…) va fuori, come chiede la guida. Un prompt già strutturato
(contiene i campi ufficiali) passa invariato.
"""
from __future__ import annotations

from core import minimax_h3_spec as S
from core import voice_markup as VM

_FIELDS = ("integrated_multimodal_description:", "detailed_description:", "subject_definitions:")
_AMBIENT = "Natural ambient sound consistent with the scene, with soft room tone."


def is_structured(prompt: str) -> bool:
    return any(f in (prompt or "") for f in _FIELDS)


def _speakers(lines: list[dict]) -> dict[str, int]:
    ids: dict[str, int] = {}
    for d in lines:
        key = (d.get("speaker") or "").strip().lower() or "_"
        ids.setdefault(key, len(ids) + 1)
    return ids


def _line_sentence(d: dict, sid: int, subject: str = "") -> str:
    text = VM.plain(d.get("text") or "").strip()
    if text and text[-1] not in ".?!":
        text += "."
    lang = d.get("language") or "English"
    who = (d.get("speaker") or "").strip()
    head = f"{subject} (S{sid})" if subject else (f"{who} (S{sid})" if who else f"The speaker (S{sid})")
    how = VM.delivery_en(d.get("text") or "")
    verb = "says" + (f", {how}," if how else "")
    return f"{head} {verb} <d>[{lang}] {text}</d>"


def compose(p: dict, duration_s: float) -> str:
    """Prompt finale dai params del worker (prompt, h3_dialogue, h3_soundscape, h3_music, image_path,
    last_image_path, references)."""
    prompt = (p.get("prompt") or "").strip()
    if is_structured(prompt):
        return prompt
    lines = [d for d in (p.get("h3_dialogue") or []) if (d.get("text") or "").strip()]
    sids = _speakers(lines)
    sound = (p.get("h3_soundscape") or "").strip() or _AMBIENT
    music = (p.get("h3_music") or "").strip() or "N/A"
    refs = p.get("references") or []
    if refs:
        return _compose_ref(prompt, lines, sids, sound, music, refs)
    head = ""
    if p.get("image_path") and p.get("last_image_path"):
        head = ("How the reference pictures align with the target video — Picture 1 (from Shot 1) aligns with "
                f"the 0.00-second mark of the target video; Picture 2 (from Shot 1) aligns with the "
                f"{duration_s:.2f}-second mark of the target video.")
    elif p.get("image_path"):
        head = "For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced."
    elif p.get("last_image_path"):
        head = ("How the reference pictures align with the target video — <Picture 1> (from [Shot 1]) aligns "
                f"with the {duration_s:.2f}-second mark of the target video.")
    body = prompt or "A person talks to the camera."
    if p.get("image_path") and "<Picture 1>" not in body:
        body = f"Live-action, cinematic, the scene shown in <Picture 1>, preserving the subjects' appearance, clothing and setting. {body}"
    if body[-1] not in ".?!":
        body += "."
    desc = "[Shot 1] " + body + (" " + " ".join(_line_sentence(d, sids[(d.get('speaker') or '').strip().lower() or '_'])
                                             for d in lines) if lines else "")
    parts = ([head, ""] if head else []) + [f"integrated_multimodal_description: {desc}", "",
                                            f"overall_soundscape: {sound}", "", f"non_diegetic_music: {music}"]
    return "\n".join(parts)


def _compose_ref(prompt, lines, sids, sound, music, refs) -> str:
    defs, keep = [], []
    subj_n = pic_n = vid_n = aud_n = 0
    first_subject = ""
    for r in refs:
        kind, desc, role = r.get("kind") or "image", (r.get("desc") or "").strip(), (r.get("role") or "").strip()
        if kind == "image":
            pic_n += 1
            subj_n += 1
            label = f"<Subject {subj_n}>"
            first_subject = first_subject or label
            what = desc or role or "the person"
            defs.append(f"{label} is {what} in <Picture {pic_n}>.")
            keep.append(f"{label} (appears in [Shot 1]): fully_preserved - identity, appearance and clothing are retained.")
        elif kind == "video":
            vid_n += 1
            subj_n += 1
            label = f"<Subject {subj_n}>"
            defs.append(f"{label} is {desc or role or 'the subject'} in <Video {vid_n}>.")
            keep.append(f"{label} (appears in [Shot 1]): fully_preserved - appearance and motion are retained.")
        elif kind == "audio":
            aud_n += 1
            target = first_subject or "the speaker"
            defs.append(f"<Audio {aud_n}> is the voice-timbre reference for {target} (S1).")
            keep.append(f"<Audio {aud_n}>: reference - the speaker follows <Audio {aud_n}>'s voice timbre without "
                        "copying the original signal.")
    tasks = ["reference generation"] + (["audio reference"] if aud_n else [])
    body = prompt or "The subject talks to the camera."
    if body[-1] not in ".?!":
        body += "."
    subj = first_subject
    talk = " ".join(_line_sentence(d, sids[(d.get("speaker") or "").strip().lower() or "_"],
                                   subj if (sids[(d.get('speaker') or '').strip().lower() or '_'] == 1 and subj) else "")
                    for d in lines)
    if aud_n and subj:
        talk = talk.replace(f"{subj} (S1) says", f"{subj} (S1), using the voice timbre referenced from <Audio 1>, says")
    summary = f"[{' + '.join(tasks)}] The target video shows {subj or 'the subject'}: {body}"
    return "\n".join(["subject_definitions:", *defs, "", "summary:", summary, "", "retention_analysis:", *keep, "",
                      "detailed_description:", f"[Shot 1] {body} {talk}".strip(), "",
                      f"overall_soundscape: {sound}", "", f"non_diegetic_music: {music}"])
