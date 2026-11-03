"""Prompt strutturati MiniMax H3 per l'EDIT con maschera (cambia sfondo / sostituisci personaggio) — modulo puro, senza Qt/torch.

Formato (subject_definitions / summary / retention_analysis / detailed_description / overall_soundscape) adattato dai prompt MIT di
Openjutsu / comfyui-h3-oneclick-swap (D3voz): vedi NOTICE-openjutsu.md. Una sola persona e una sola foto (fase 1); la lista dei piani
descritta da un VLM arriva con la fase 2 (qui `shots` = testo libero dell'utente o un default neutro).
"""
from __future__ import annotations

FPS = 24

SOUND = """overall_soundscape:
Any ambience and physical sounds already present in <Audio 1> remain part of its unchanged mix. Add no new room noise, footsteps, voices or impacts.

non_diegetic_music:
Reuse the original music and vocals in <Audio 1> at their original tempo, pitch, mix and synchronization. Do not generate a new performance or additional score."""

LIP_SYNC = ("The visible people lip-sync any vocals in <Audio 1> where the source people sing or speak; "
            "the soundtrack itself is unchanged. ")
NO_LIP_SYNC = ("Nobody on screen sings, speaks or lip-syncs; the music in <Audio 1> is a background track added later. "
               "Mouths stay closed or keep the source expressions such as smiles; the soundtrack itself is unchanged. ")
ALIGN_SENTENCE = ("<Video 1> is frame-accurate: frame k of the target video shows exactly the moment of frame k of <Video 1>; the person's steps, head and "
                  "mouth must match <Video 1> frame by frame, with no delay.")
DEFAULT_SHOTS = "The framing, shot sizes and subject positions follow <Video 1> in every shot."


def _first_upper(text: str) -> str:
    return text[:1].upper() + text[1:]


def _sentence(text: str) -> str:
    text = " ".join((text or "").split())
    return text if not text or text[-1] in ".!?" else text + "."


def background_prompt(setting: str, subject: str, length_frames: int, overlap_frames: int, *, has_picture: bool = False,
                      picture_desc: str = "", shots: str = "", lip_sync: bool = True, variant: str = "base") -> str:
    """CAMBIA SFONDO: `subject` = chi si tiene (es. «the woman»), `setting` = il nuovo luogo a parole. Nel video di riferimento
    (<Video 1>) lo sfondo e una sfocatura grigia liscia: porta solo il movimento di camera e la disposizione grossolana.
    `variant` (ABLAZIONE 03/11, un fattore alla volta; "base" = il prompt di sempre):
      · no_performance: senza «keeps their exact look and performance» e senza la retention «the actions ... are retained» (si descrive solo il nuovo ambiente);
      · replace_like:   come il prompt di Sostituisci: la persona «segue i tempi della performance della figura» e compie le stesse azioni agli stessi istanti, senza «exact look»;
      · carryover:      il prompt di sempre (il chiamante passa overlap_frames > 0 anche senza contesto: la frase «Its first N seconds are already-edited footage» da sola);
      · align:          il prompt di sempre + la frase esplicita di allineamento temporale (ALIGN_SENTENCE) dopo la definizione di <Video 1>;
      · carryover_align: le due cose insieme."""
    sec = length_frames / FPS
    new = "<Subject 1>" + (", " + _sentence(setting).rstrip(".") if setting and has_picture else "")
    lines = ["subject_definitions:"]
    if has_picture:
        lines.append(("<Subject 1> is the new setting from <Picture 1>. %s" % _sentence(picture_desc or setting)).strip())
    else:
        lines.append("<Subject 1> is the new setting: %s" % (_sentence(setting) or "a new location, unlike the original."))
    lines += [
        "<Video 1> is the source footage being edited, a %.3f-second section at 24 fps, in which the setting around %s is shown as a smooth "
        "grey blur that supplies only the camera movement and rough layout. It supplies the exact camera framing, camera movement, people, "
        "performance timing and hard cuts." % (sec, subject),
        *([ALIGN_SENTENCE] if "align" in variant else []),
        "<Audio 1> is the synchronized original soundtrack supplied with <Video 1>, reused unchanged as the complete final soundtrack.",
        "", "summary:",
        "[video editing + reference generation + audio reuse] The target video is an edited version of <Video 1>. %s" % _bg_summary(variant, subject, new)
        + ("Its first %.3f seconds are already-edited footage carried over from the preceding section. " % (overlap_frames / FPS)
           if overlap_frames else "") + "Everything else keeps its source look, timing and layout.",
        "", "retention_analysis:",
        "<Subject 1> (setting): partially_preserved - the look, materials and lighting of the new setting come from it; the camera framing and "
        "movement are those of <Video 1>.",
        _bg_retention(variant, subject),
        "<Audio 1>: fully_copy - the original synchronized soundtrack is the complete final audio, without additional generated vocals or "
        "sound effects.",
        "", "detailed_description:", "Live-action footage with the source lighting and lens perspective.",
        shots.strip() or DEFAULT_SHOTS,
        "Throughout all shots, the surroundings are %s, lit to match %s; no part of the original location remains." % (new, subject),
        ("%s performs every action of the figure in <Video 1>: the same gestures, steps, gaze, expressions and mouth shapes at the same moments. " % _first_upper(subject)
         if variant == "replace_like" else "")
        + "Continue the original camera movement and performance through the last frame without adding transitions or changing speed. "
        + (LIP_SYNC if lip_sync else NO_LIP_SYNC),
        "", SOUND,
    ]
    return "\n".join(lines)


def _bg_summary(variant: str, subject: str, new: str) -> str:
    if variant == "no_performance":
        return "The setting around the person becomes %s. " % new
    if variant == "replace_like":
        return "%s follows the timing of the performance of the figure in <Video 1>, while the setting around them becomes %s. " % (_first_upper(subject), new)
    return "%s keeps their exact look and performance, while the setting around them becomes %s. " % (_first_upper(subject), new)


def _bg_retention(variant: str, subject: str) -> str:
    if variant == "no_performance":
        return "<Video 1> (edited source): partially_preserved - only the setting changes; the camera movement, cuts and timing are retained."
    if variant == "replace_like":
        return ("<Video 1> (edited source): partially_preserved - only the setting changes; camera movement, shot composition, actions, interaction, "
                "cuts and timing are retained.")
    return "<Video 1> (edited source): partially_preserved - only the setting changes; %s, the actions, camera movement, cuts and timing are retained." % subject


def background_empty_person_prompt(setting: str, length_frames: int, overlap_frames: int, *, has_picture: bool = False,
                                   picture_desc: str = "", shots: str = "") -> str:
    """CAMBIA SFONDO, variante «zona della persona vuota» (bg_prompt_mode=empty_person, attiva con bg_person_ref=erase): nel riferimento (<Video 1>) non c'e
    nessuna persona, solo l'ambiente sfocato; la persona vera e gia nel contenuto tenuto e viene ricomposta dopo. Il modello genera SOLO l'ambiente, coerente
    con camera e luce, e non disegna ne ridisegna nessuna persona: niente lip sync."""
    sec = length_frames / FPS
    lines = ["subject_definitions:"]
    if has_picture:
        lines.append(("<Subject 1> is the new setting from <Picture 1>. %s" % _sentence(picture_desc or setting)).strip())
    else:
        lines.append("<Subject 1> is the new setting: %s" % (_sentence(setting) or "a new location, unlike the original."))
    lines += [
        "<Video 1> is the source footage, a %.3f-second section at 24 fps, shown as a smooth blur of the surroundings with no person in it: it "
        "supplies only the exact camera framing, camera movement, light direction and rough layout of the environment." % sec,
        "<Audio 1> is the synchronized original soundtrack supplied with <Video 1>, reused unchanged as the complete final soundtrack.",
        "", "summary:",
        "[video editing + environment generation + audio reuse] The target video shows only the new environment, <Subject 1>, seen through the "
        "camera of <Video 1>. The area where a person would stand is empty space: no person, figure, silhouette or double is drawn there, "
        "because the real person is composed over it afterwards. Footage already present in the content is kept exactly as given. "
        + ("Its first %.3f seconds are already-edited footage carried over from the preceding section. " % (overlap_frames / FPS)
           if overlap_frames else ""),
        "", "retention_analysis:",
        "<Subject 1> (setting): partially_preserved - the look, materials and lighting of the new setting come from it; the camera framing and "
        "movement are those of <Video 1>.",
        "<Video 1> (camera reference): partially_preserved - only the camera framing, movement, cuts and timing are retained; it contains no person to copy.",
        "<Audio 1>: fully_copy - the original synchronized soundtrack is the complete final audio, without additional generated vocals or "
        "sound effects.",
        "", "detailed_description:", "Live-action footage of an environment with the source lighting and lens perspective, lit consistently with the camera.",
        shots.strip() or "The framing and camera movement follow <Video 1> in every shot.",
        "Throughout all shots only the environment is generated; add no person, face, body, shadow of a person or any new figure anywhere, "
        "and do not redraw or duplicate any person already present in the carried-over content. " + NO_LIP_SYNC,
        "", SOUND,
    ]
    return "\n".join(lines)


def replace_prompt(role: str, length_frames: int, overlap_frames: int, *, picture_desc: str = "", shots: str = "",
                   lip_sync: bool = True) -> str:
    """SOSTITUISCI PERSONAGGIO (una persona): `role` = chi si sostituisce (es. «the woman in the red dress»); <Picture 1> = il nuovo personaggio;
    <Video 1> = il sorgente in cui la persona e una figura grigia liscia (solo posa e movimento)."""
    sec = length_frames / FPS
    desc = _sentence(picture_desc)
    lines = [
        "subject_definitions:",
        ("<Subject 1> is the person from <Picture 1>. %s They replace %s in <Video 1>, in every shot where that person appears. "
         "Their face, hair, body and clothing all come from <Picture 1>." % (desc, role)).replace("  ", " "),
        "<Subject 2> is the setting, props and lighting in <Video 1>.",
        "<Video 1> is the source footage being edited, a %.3f-second section at 24 fps, in which the person to replace is shown as a smooth grey "
        "figure. It supplies the exact camera framing, camera movement, performance timing, body poses, interactions and hard cuts. "
        "The grey figure supplies movement only; faces, hair, skin and clothing come from the picture." % sec,
        "<Audio 1> is the synchronized original soundtrack supplied with <Video 1>, reused unchanged as the complete final soundtrack.",
        "", "summary:",
        "[video editing + reference generation + audio reuse] The target video is an edited version of <Video 1>. <Subject 1> takes the place of %s. "
        "Identity and complete outfit come from the picture. " % role
        + ("Its first %.3f seconds are already-edited footage carried over from the preceding section. " % (overlap_frames / FPS)
           if overlap_frames else "")
        + "The source performance, camera, cuts, setting, props and soundtrack retain their timing and layout.",
        "", "retention_analysis:",
        "<Subject 1> (all shots): fully_preserved - the referenced facial identity, hair, skin tone and complete outfit remain consistent while "
        "poses and mouth movements follow %s." % role,
        "<Subject 2> (setting and props): fully_preserved - the location, props and lighting retain their source appearance and position.",
        "<Video 1> (edited source): partially_preserved - the replaced person's identity and outfit change; camera movement, shot composition, "
        "actions, interaction, cuts and timing are retained.",
        "<Audio 1>: fully_copy - the original synchronized soundtrack is the complete final audio, without additional generated vocals or "
        "sound effects.",
        "", "detailed_description:",
        "Live-action footage in the same setting as <Video 1>, with natural skin tones and the source lighting and lens perspective. "
        "The replacement person appears only where the person they replace is visible; the background of the picture does not enter the scene.",
        shots.strip() or DEFAULT_SHOTS,
        "<Subject 1> performs every action of %s: the same gestures, steps, gaze, expressions and mouth shapes at the same moments." % role,
        "Throughout all shots, %s is always the person in <Picture 1>, regardless of screen position or size. Their outfit replaces the original "
        "clothing completely; no face, hair, skin, costume, fabric pattern or accessory of %s remains. %sContinue the original camera "
        "movement and performance through the last frame without adding transitions or changing speed." % (role, role, LIP_SYNC if lip_sync else NO_LIP_SYNC),
        "", SOUND,
    ]
    return "\n".join(lines)
