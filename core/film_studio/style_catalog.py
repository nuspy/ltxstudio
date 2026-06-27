"""Cataloghi stili (~50) e generi (~20) + linguaggio di camera per l'AI Film Studio.
Liste editabili; usate nei menu del tab Input/Scenografia e iniettate nei prompt."""
from __future__ import annotations

STYLES = [
    "photorealistic", "cinematic", "hyperrealistic", "documentary", "film noir", "black and white",
    "sepia vintage", "technicolor", "pastel", "high contrast", "low key", "high key", "golden hour",
    "blue hour", "neon cyberpunk", "synthwave", "vaporwave", "surrealist", "minimalist", "maximalist",
    "expressionist", "impressionist", "watercolor", "oil painting", "pencil sketch", "charcoal",
    "ink illustration", "comic book", "graphic novel", "manga", "anime", "cel-shaded cartoon",
    "3D pixar style", "claymation", "stop motion", "pixel art", "low poly", "isometric",
    "baroque", "renaissance", "gothic", "art deco", "art nouveau", "bauhaus", "brutalist",
    "steampunk", "dieselpunk", "fantasy epic", "fairytale storybook", "horror gothic", "grindhouse",
    "western", "retro 80s", "vintage 70s film grain", "polaroid", "lomography", "infrared",
    "thermal", "double exposure", "tilt-shift miniature", "macro detail",
]

GENRES = [
    "drama", "comedy", "romance", "thriller", "horror", "science fiction", "fantasy", "action",
    "adventure", "mystery", "crime", "noir", "war", "western", "musical", "documentary",
    "animation", "family", "biopic", "historical", "psychological", "dark comedy", "satire",
]

# linguaggio di camera (tag suggeriti dalla LLM / selezionabili)
SHOT_TYPES = ["extreme wide shot", "wide shot", "full shot", "medium long shot", "medium shot",
              "medium close-up", "close-up", "extreme close-up", "over-the-shoulder", "POV",
              "two-shot", "establishing shot", "insert/detail", "aerial", "low angle", "high angle",
              "dutch angle", "eye level"]
CAMERA_MOVES = ["static", "pan left", "pan right", "tilt up", "tilt down", "dolly in", "dolly out",
                "tracking", "crane up", "crane down", "orbit", "handheld", "steadicam", "zoom in",
                "zoom out", "whip pan"]
TRANSITIONS = ["cut", "dissolve", "fade in", "fade out", "fade to black", "wipe", "match cut",
               "smash cut", "cross dissolve", "iris"]
