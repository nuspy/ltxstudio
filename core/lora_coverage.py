"""Copertura automatica best-practice per dataset LoRA video (LTX-2.3).

Per ogni TIPO di soggetto definisce i "casi" da generare (vista/espressione/luce/movimento…),
ognuno con: id, template di prompt (con {subj} = descrizione del soggetto) e tag (entrano nella
caption assieme al trigger). L'obiettivo è una copertura ampia e variata = LoRA robusta.

Un "caso" → un clip di training. L'utente può attivare/disattivare i casi e aggiungere
"situazioni custom" (descrizioni libere) in numero illimitato.
"""
from __future__ import annotations

# ── PERSONAGGIO: viste × espressioni/emozioni × inquadrature × luci (set curato, non prodotto pieno) ──
PERSON_CASES = [
    # viste base / neutro
    ("front_neutral",   "{subj}, front view, neutral calm expression, looking straight at the camera", ["front view", "neutral expression"]),
    ("three_quarter_l", "{subj}, three-quarter left view, subtle natural expression",                  ["three-quarter view", "left"]),
    ("three_quarter_r", "{subj}, three-quarter right view, subtle natural expression",                 ["three-quarter view", "right"]),
    ("profile",         "{subj}, side profile view, calm expression",                                  ["profile view"]),
    ("low_angle",       "{subj}, slight low-angle shot, confident look",                               ["low angle"]),
    ("high_angle",      "{subj}, slight high-angle shot, looking up",                                  ["high angle"]),
    # espressioni / emozioni
    ("smile",           "{subj}, warm genuine smile, friendly",                                        ["smiling"]),
    ("laugh",           "{subj}, big joyful laugh, eyes crinkled",                                      ["laughing"]),
    ("sad",             "{subj}, sad sorrowful expression, downcast eyes",                              ["sad expression"]),
    ("angry",           "{subj}, angry furious expression, frowning",                                  ["angry expression"]),
    ("surprised",       "{subj}, surprised shocked expression, wide eyes, open mouth",                  ["surprised", "shocked"]),
    ("shout",           "{subj}, shouting loudly, mouth wide open, intense",                            ["shouting"]),
    ("fear",            "{subj}, fearful scared expression, tense",                                     ["fearful"]),
    ("grimace",         "{subj}, funny grimace, scrunched face",                                        ["grimace"]),
    ("smirk",           "{subj}, mocking smirk, raised eyebrow, sly look",                              ["smirk", "mocking"]),
    ("crying",          "{subj}, crying with tears, emotional",                                         ["crying"]),
    ("wink",            "{subj}, playful wink and slight smile",                                        ["winking"]),
    # gesti / dinamica
    ("talking",         "{subj}, talking naturally, gentle head movements",                            ["talking"]),
    ("turn_head",       "{subj}, slowly turning the head from side to side",                            ["turning head"]),
    ("nod",             "{subj}, nodding, expressive face",                                             ["nodding"]),
    # inquadrature / luci
    ("closeup",         "{subj}, extreme close-up portrait, sharp facial detail",                      ["close-up"]),
    ("medium_shot",     "{subj}, medium shot from chest up, soft daylight",                            ["medium shot", "daylight"]),
    ("full_body",       "{subj}, full body shot, standing, natural pose",                              ["full body"]),
    ("golden_hour",     "{subj}, portrait in warm golden-hour light",                                  ["golden hour"]),
    ("indoor_warm",     "{subj}, indoor warm soft lighting, cozy",                                     ["indoor", "warm light"]),
    ("dramatic",        "{subj}, dramatic cinematic lighting, strong contrast",                        ["dramatic lighting"]),
]

# ── LUOGO / ARCHITETTURA / AMBIENTE: orari-luci × viste × movimenti camera ──
PLACE_CASES = [
    ("wide_day",        "{subj}, wide establishing shot, clear midday light",                          ["wide shot", "daytime"]),
    ("wide_dawn",       "{subj}, wide shot at dawn, soft pink light",                                   ["wide shot", "dawn"]),
    ("golden_hour",     "{subj}, warm golden-hour light, long shadows",                                 ["golden hour"]),
    ("blue_hour",       "{subj}, blue-hour twilight, cool tones",                                       ["blue hour", "twilight"]),
    ("night",           "{subj}, at night, artificial lights glowing",                                  ["night"]),
    ("overcast",        "{subj}, soft overcast diffuse light",                                          ["overcast"]),
    ("foggy",           "{subj}, atmospheric fog, moody",                                               ["fog"]),
    ("rain",            "{subj}, light rain, wet reflective surfaces",                                  ["rain"]),
    ("interior",        "{subj}, interior view, natural light from windows",                            ["interior"]),
    ("detail",          "{subj}, close detail shot of materials and textures",                         ["detail", "texture"]),
    ("aerial",          "{subj}, high aerial overview shot",                                            ["aerial", "high angle"]),
    ("eye_level",       "{subj}, eye-level walkthrough perspective",                                    ["eye level", "walkthrough"]),
    ("pan",             "{subj}, slow cinematic camera pan across the scene",                           ["camera pan"]),
    ("dolly",           "{subj}, slow dolly forward into the scene",                                    ["dolly in"]),
    ("orbit",           "{subj}, camera slowly orbiting the scene",                                     ["orbit"]),
]

# ── OGGETTO: giro 360° × luci × dettaglio × contesto ──
OBJECT_CASES = [
    ("front",           "{subj}, front view, studio lighting, plain background",                        ["front view", "studio"]),
    ("side_l",          "{subj}, left side view, studio lighting",                                      ["side view", "left"]),
    ("side_r",          "{subj}, right side view, studio lighting",                                     ["side view", "right"]),
    ("back",            "{subj}, back view, studio lighting",                                           ["back view"]),
    ("top",             "{subj}, top-down view",                                                        ["top view"]),
    ("turntable",       "{subj}, slow 360-degree turntable rotation, studio lighting",                 ["turntable", "rotation"]),
    ("detail",          "{subj}, extreme close-up of material and surface detail",                      ["detail", "macro"]),
    ("dramatic",        "{subj}, dramatic rim lighting, dark background",                               ["dramatic lighting"]),
    ("natural",         "{subj}, natural daylight, soft shadows",                                       ["natural light"]),
    ("in_context",      "{subj}, shown in a realistic everyday context/environment",                    ["in context"]),
]

# angolo preferito per caso personaggio → sceglie la foto di riferimento giusta (se multi-foto)
PERSON_CASE_ANGLE = {
    "front_neutral": "front", "smile": "front", "laugh": "front", "sad": "front", "angry": "front",
    "surprised": "front", "shout": "front", "fear": "front", "grimace": "front", "smirk": "front",
    "crying": "front", "wink": "front", "talking": "front", "nod": "front", "closeup": "front",
    "three_quarter_l": "three_quarter", "three_quarter_r": "three_quarter",
    "profile": "profile", "low_angle": "front", "high_angle": "front",
    "turn_head": "three_quarter", "medium_shot": "front", "full_body": "front",
    "golden_hour": "front", "indoor_warm": "front", "dramatic": "front",
}
# clip di ROTAZIONE multi-keyframe (usano più foto/angoli in sequenza per coerenza 3D)
ROTATION_CASE = {"id": "rotation", "tags": ["turntable", "multiple angles"],
                 "prompt": "{subj}, slow consistent rotation showing front, three-quarter and side views"}

# Per FOCUS VOLTO: si variano abiti e sfondo ad ogni clip (l'unico invariante resta il viso) e si
# mettono in caption (così sono attribuiti a queste parole, non al trigger). Pool ampio = robustezza.
OUTFITS = ["a plain white t-shirt", "a black hoodie", "a denim jacket", "a formal suit", "a knitted sweater",
           "a summer dress", "a leather jacket", "a casual flannel shirt", "a sportswear top", "a wool coat",
           "a striped shirt", "a turtleneck"]
BACKDROPS = ["a neutral grey studio background", "a city street", "a green park", "an indoor room",
             "a cafe interior", "a beach at daytime", "a plain dark background", "a brick wall",
             "a forest", "an office", "a kitchen", "a sunset sky"]

CASES = {"person": PERSON_CASES, "place": PLACE_CASES, "object": OBJECT_CASES, "concept": []}
# default attivi (un sottoinsieme ben bilanciato; l'utente può cambiarli)
DEFAULTS = {
    "person": ["front_neutral", "three_quarter_l", "three_quarter_r", "profile", "smile", "laugh",
               "sad", "angry", "surprised", "shout", "grimace", "smirk", "talking", "closeup",
               "medium_shot", "full_body", "golden_hour", "dramatic"],
    "place":  ["wide_day", "wide_dawn", "golden_hour", "blue_hour", "night", "overcast",
               "interior", "detail", "aerial", "eye_level", "pan", "dolly"],
    "object": ["front", "side_l", "side_r", "back", "turntable", "detail", "dramatic", "natural", "in_context"],
    "concept": [],
}
TYPES = ["person", "place", "object", "concept"]


def cases_for(subject_type: str) -> list[tuple]:
    return CASES.get(subject_type, [])


def default_case_ids(subject_type: str) -> list[str]:
    return list(DEFAULTS.get(subject_type, []))
