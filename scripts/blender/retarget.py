"""Retarget di un'animazione mocap (BVH) su un modello 3D riggato (eseguito in
Blender headless).

Regola chiave (richiesta utente): se il modello ha GIÀ un'armatura (output del
RigTab), l'animazione viene applicata a QUELLA armatura — NON si aggiunge un
secondo scheletro. Solo se il modello è privo di rig si tiene l'armatura del BVH
(fallback) per non perdere il movimento.

Args JSON (dopo `--`):
  {"model": "<glb/fbx riggato>", "anim": "<file .bvh>", "output": "<path>",
   "format": "fbx"|"glb"|"bvh", "template": "Mixamo"|"Unreal Engine 5"|"auto",
   "mapping": {"<osso_bvh>": "<osso_target>", ...},  # opzionale, vince sul template
   "frame_start": int, "frame_end": int}
Emette `@@RESULT@@{"ok": true, "output": "<path>", "retargeted": bool}`.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bl_common as B  # noqa: E402
import bpy  # type: ignore  # noqa: E402


# Mappa di default: nomi ossa BVH "umanoidi standard" → nomi del template.
# I BVH da MediaPipe/FreeMoCap usano nomi tipo Hips/Spine/LeftArm/...; i rig
# SkinTokens con template Mixamo/UE5 usano i naming noti qui sotto.
_MIXAMO = {
    "Hips": "mixamorig:Hips", "Spine": "mixamorig:Spine", "Spine1": "mixamorig:Spine1",
    "Spine2": "mixamorig:Spine2", "Neck": "mixamorig:Neck", "Head": "mixamorig:Head",
    "LeftShoulder": "mixamorig:LeftShoulder", "LeftArm": "mixamorig:LeftArm",
    "LeftForeArm": "mixamorig:LeftForeArm", "LeftHand": "mixamorig:LeftHand",
    "RightShoulder": "mixamorig:RightShoulder", "RightArm": "mixamorig:RightArm",
    "RightForeArm": "mixamorig:RightForeArm", "RightHand": "mixamorig:RightHand",
    "LeftUpLeg": "mixamorig:LeftUpLeg", "LeftLeg": "mixamorig:LeftLeg",
    "LeftFoot": "mixamorig:LeftFoot", "RightUpLeg": "mixamorig:RightUpLeg",
    "RightLeg": "mixamorig:RightLeg", "RightFoot": "mixamorig:RightFoot",
}
_UE5 = {
    "Hips": "pelvis", "Spine": "spine_01", "Spine1": "spine_03", "Spine2": "spine_05",
    "Neck": "neck_01", "Head": "head",
    "LeftShoulder": "clavicle_l", "LeftArm": "upperarm_l", "LeftForeArm": "lowerarm_l",
    "LeftHand": "hand_l", "RightShoulder": "clavicle_r", "RightArm": "upperarm_r",
    "RightForeArm": "lowerarm_r", "RightHand": "hand_r",
    "LeftUpLeg": "thigh_l", "LeftLeg": "calf_l", "LeftFoot": "foot_l",
    "RightUpLeg": "thigh_r", "RightLeg": "calf_r", "RightFoot": "foot_r",
}


def _mapping_for(template: str) -> dict:
    t = (template or "auto").lower()
    if "mixamo" in t:
        return _MIXAMO
    if "unreal" in t or "ue5" in t:
        return _UE5
    return {}   # "auto"/identità: prova i nomi BVH così come sono


def _target_armature(exclude=None):
    for o in bpy.data.objects:
        if o.type == "ARMATURE" and o is not exclude:
            return o
    return None


def _retarget(target, source, mapping, fstart, fend) -> None:
    """Constraint Copy Rotation/Location source→target, poi bake e pulizia."""
    bpy.context.scene.frame_start = fstart
    bpy.context.scene.frame_end = fend

    # identità se mapping vuoto: usa i nomi delle ossa del source presenti nel target
    if not mapping:
        mapping = {b.name: b.name for b in source.pose.bones
                   if b.name in target.pose.bones}

    root_done = False
    for src_name, tgt_name in mapping.items():
        tb = target.pose.bones.get(tgt_name)
        if tb is None or src_name not in source.pose.bones:
            continue
        rot = tb.constraints.new("COPY_ROTATION")
        rot.target = source
        rot.subtarget = src_name
        # La sola radice (hips/pelvis) copia anche la posizione, per lo spostamento.
        if not root_done and ("hip" in tgt_name.lower() or "pelvis" in tgt_name.lower()):
            loc = tb.constraints.new("COPY_LOCATION")
            loc.target = source
            loc.subtarget = src_name
            root_done = True

    # Bake dell'animazione sul target con visual keying, poi rimuovi i constraint.
    for o in bpy.data.objects:
        o.select_set(False)
    target.select_set(True)
    bpy.context.view_layer.objects.active = target
    # only_selected=False: per bake_types POSE 'only_selected' filtra le OSSA selezionate (non l'oggetto).
    # Selezionavamo solo l'oggetto target → 0 ossa selezionate → bake vuoto. False = tutte le ossa.
    bpy.ops.nla.bake(frame_start=fstart, frame_end=fend, only_selected=False,
                     visual_keying=True, clear_constraints=True,
                     use_current_action=True, bake_types={"POSE"})


def main() -> None:
    try:
        args = B.parse_args()
        model = args["model"]
        anim = args["anim"]
        if not os.path.isfile(model):
            raise RuntimeError(f"Modello inesistente: {model}")
        if not os.path.isfile(anim):
            raise RuntimeError(f"File animazione inesistente: {anim}")

        B.clear_scene()
        B.import_file(model)
        target = _target_armature()           # armatura del rig (se presente)

        B.import_file(anim)                    # crea l'armatura sorgente col BVH
        source = _target_armature(exclude=target)
        if source is None:
            raise RuntimeError("Il BVH non ha prodotto un'armatura sorgente.")

        # frame range dall'azione del BVH
        fstart = int(args.get("frame_start", bpy.context.scene.frame_start))
        fend = int(args.get("frame_end", 0)) or int(bpy.context.scene.frame_end)
        if source.animation_data and source.animation_data.action:
            fr = source.animation_data.action.frame_range
            fstart, fend = int(fr[0]), int(fr[1])

        retargeted = False
        if target is not None:
            # ARMATURA ESISTENTE → applica l'animazione a QUELLA (no nuovo scheletro)
            mapping = args.get("mapping") or _mapping_for(args.get("template", "auto"))
            _retarget(target, source, mapping, fstart, fend)
            # rimuovi l'armatura sorgente del BVH (non serve più)
            bpy.data.objects.remove(source, do_unlink=True)
            retargeted = True
        # else: nessun rig nel modello → si tiene l'armatura del BVH (fallback addon).

        out = args.get("output") or os.path.splitext(model)[0] + "_anim.fbx"
        fmt = args.get("format") or os.path.splitext(out)[1].lstrip(".") or "fbx"
        final = B.export_scene(out, fmt, {"animations": True})
        B.emit_result(True, output=final, retargeted=retargeted)
    except Exception as e:  # noqa: BLE001
        B.emit_result(False, error=str(e))


main()
