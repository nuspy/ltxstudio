"""Workflow predefiniti per LTX Studio.

Ogni workflow è un profilo che imposta tutti i parametri GUI principali + un prompt template.
Selezionando un workflow dal dropdown si pre-compilano automaticamente tutti i campi.

Categorie:
- Persone & Avatar (talking head, singing, portrait, interview, …)
- Paesaggi & Natura (cinematic landscape, timelapse, underwater, …)
- Prodotti & Commerciale (product showcase, food, fashion, …)
- Azione & Sport (action scene, martial arts, dance, …)
- Fantasy & Sci-Fi (fantasy, cyberpunk, space, …)
- Animazione & Arte (animation, painting, abstract, …)
- Architettura & Interni (real estate, architecture tour, …)
- Horror & Dark (horror, noir, gothic, …)
- Documentario & News (interview, news, documentary, …)
- Speciali (micro close-up, drone shot, underwater, …)
"""
from __future__ import annotations

WORKFLOW_PRESETS: dict[str, dict] = {

    # ══════════════════════════════════════════════════════════════════
    # PERSONE & AVATAR
    # ══════════════════════════════════════════════════════════════════

    "👤 Talking Head (presentatore)": {
        "steps": 30, "cfg": 2.5, "sampler": "euler_ancestral_cfg_pp",
        "scheduler": "beta", "width": 704, "height": 1280, "fps": 25, "length_s": 8.0,
        "nsfw_mode": "sfw", "audio_mode": "native",
        "prompt_template": "portrait of a {subject} speaking directly to camera, professional presenter, "
                           "clean background, studio lighting, sharp focus, talking, natural lip movements, "
                           "professional attire, 4K, photorealistic",
        "notes": "Ottimale per video tutorial, spiegazioni, presentazioni. Carica audio con modalità 'Voce clone'.",
        "lora_suggestions": ["talking_head", "face_detail"],
    },

    "🎤 Singing Avatar": {
        "steps": 30, "cfg": 2.5, "sampler": "euler_ancestral_cfg_pp",
        "scheduler": "beta", "width": 704, "height": 1280, "fps": 24, "length_s": 8.0,
        "nsfw_mode": "sfw", "audio_mode": "clone",
        "prompt_template": "portrait of a {subject} singing, microphone, performance stage, "
                           "concert lighting, dynamic pose, open mouth, expressive face, "
                           "music performance, 4K",
        "notes": "Abbina audio clone con campione vocale. Abilita ControlNet (Pose) per movimenti realistici.",
        "lora_suggestions": ["singing_motion", "portrait_detail"],
    },

    "🎭 Attore — Primo Piano": {
        "steps": 35, "cfg": 3.0, "sampler": "dpmpp_2m_sde",
        "scheduler": "exponential", "width": 896, "height": 1152, "fps": 24, "length_s": 6.0,
        "nsfw_mode": "sfw", "audio_mode": "native",
        "prompt_template": "cinematic close-up of {subject}, dramatic expression, film grain, "
                           "cinematic lighting, shallow depth of field, movie scene, "
                           "award-winning cinematography, 8K",
        "notes": "Ideale per scene drammatiche, monologhi, close-up emotivi.",
        "lora_suggestions": [],
    },

    "💼 Corporate Portrait": {
        "steps": 25, "cfg": 2.5, "sampler": "euler_ancestral_cfg_pp",
        "scheduler": "beta", "width": 768, "height": 1024, "fps": 25, "length_s": 5.0,
        "nsfw_mode": "sfw", "audio_mode": "native",
        "prompt_template": "professional corporate portrait of {subject}, business attire, "
                           "neutral background, confident smile, perfect lighting, LinkedIn photo style, "
                           "high quality, sharp",
        "notes": "Per foto aziendali animate, intro LinkedIn, headshots video.",
        "lora_suggestions": [],
    },

    "👩‍🎨 Portrait Artistico": {
        "steps": 45, "cfg": 4.0, "sampler": "dpmpp_2m_sde",
        "scheduler": "exponential", "width": 768, "height": 1024, "fps": 24, "length_s": 6.0,
        "nsfw_mode": "sfw", "audio_mode": "native",
        "prompt_template": "artistic portrait of {subject}, painterly style, dramatic chiaroscuro, "
                           "Renaissance lighting, rich colors, oil painting texture, masterpiece, "
                           "museum quality",
        "notes": "Stile pittorico classico. Combina con LoRA stile pittorico.",
        "lora_suggestions": ["oil_painting", "renaissance_style"],
    },

    "🤼 Doppio Presentatore": {
        "steps": 28, "cfg": 2.5, "sampler": "euler_ancestral_cfg_pp",
        "scheduler": "beta", "width": 1280, "height": 704, "fps": 25, "length_s": 10.0,
        "nsfw_mode": "sfw", "audio_mode": "native",
        "prompt_template": "two professional presenters in a studio, talking to each other, "
                           "news desk, professional lighting, 16:9 broadcast format, "
                           "natural conversation, engaged expressions",
        "notes": "Per format interviste, talk show, podcast video.",
        "lora_suggestions": [],
    },

    # ══════════════════════════════════════════════════════════════════
    # PAESAGGI & NATURA
    # ══════════════════════════════════════════════════════════════════

    "🌄 Landscape Cinematografico": {
        "steps": 40, "cfg": 3.5, "sampler": "dpmpp_2m_sde",
        "scheduler": "exponential", "width": 1280, "height": 544, "fps": 24, "length_s": 10.0,
        "nsfw_mode": "sfw", "audio_mode": "native",
        "prompt_template": "epic cinematic landscape, {subject}, golden hour lighting, "
                           "majestic mountains, sweeping vista, cinematic color grading, "
                           "anamorphic lens, IMAX quality, breathtaking, award-winning photography",
        "notes": "Formato cinemascope 2.35:1. Ottimale per intro, title cards, sfondi.",
        "lora_suggestions": ["landscape_detail", "cinematic_lut"],
    },

    "🌅 Alba / Tramonto": {
        "steps": 35, "cfg": 3.0, "sampler": "dpmpp_2m_sde",
        "scheduler": "exponential", "width": 1280, "height": 704, "fps": 24, "length_s": 8.0,
        "nsfw_mode": "sfw", "audio_mode": "native",
        "prompt_template": "stunning {subject} at {time_of_day}, warm golden light, dramatic sky, "
                           "silhouettes, lens flare, long exposure, ultra detailed, "
                           "National Geographic style",
        "notes": "Sostituisci {time_of_day} con 'sunset' o 'sunrise'.",
        "lora_suggestions": [],
    },

    "🌊 Oceano / Costa": {
        "steps": 35, "cfg": 3.0, "sampler": "dpmpp_2m_sde",
        "scheduler": "exponential", "width": 1280, "height": 704, "fps": 24, "length_s": 8.0,
        "nsfw_mode": "sfw", "audio_mode": "native",
        "prompt_template": "stunning ocean coastline, {subject}, crashing waves, sea spray, "
                           "turquoise water, dramatic cliffs, coastal atmosphere, "
                           "BBC Planet Earth quality, ultra HD",
        "notes": "Adatto per trailer, documentari naturalistici, sfondi relax.",
        "lora_suggestions": [],
    },

    "⏱️ Timelapse Urbano": {
        "steps": 30, "cfg": 3.0, "sampler": "euler_ancestral_cfg_pp",
        "scheduler": "beta", "width": 1280, "height": 704, "fps": 30, "length_s": 6.0,
        "nsfw_mode": "sfw", "audio_mode": "native",
        "prompt_template": "urban timelapse of {city}, busy streets, light trails, "
                           "day to night transition, skyscrapers, motion blur, "
                           "hyperlapse style, vivid city lights",
        "notes": "Per intro aziendali, opening video, social media reel.",
        "lora_suggestions": [],
    },

    "❄️ Paesaggio Invernale": {
        "steps": 35, "cfg": 3.0, "sampler": "dpmpp_2m_sde",
        "scheduler": "exponential", "width": 1280, "height": 704, "fps": 24, "length_s": 8.0,
        "nsfw_mode": "sfw", "audio_mode": "native",
        "prompt_template": "serene winter landscape, {subject}, fresh snow, frozen lake, "
                           "snow-covered trees, blue hour lighting, misty atmosphere, "
                           "ultra detailed, cozy winter mood",
        "notes": "Ottimale per pubblicità natalizie, sfondi meditazione.",
        "lora_suggestions": [],
    },

    "🌋 Volcano / Eruzione": {
        "steps": 40, "cfg": 3.5, "sampler": "dpmpp_2m_sde",
        "scheduler": "exponential", "width": 1280, "height": 544, "fps": 24, "length_s": 8.0,
        "nsfw_mode": "sfw", "audio_mode": "native",
        "prompt_template": "dramatic volcano eruption, {subject}, lava flows, ash cloud, "
                           "orange glow, night scene, powerful forces of nature, "
                           "BBC documentary style, ultra dramatic",
        "notes": "Scenario drammatico per documentari, giochi, trailer.",
        "lora_suggestions": [],
    },

    # ══════════════════════════════════════════════════════════════════
    # PRODOTTI & COMMERCIALE
    # ══════════════════════════════════════════════════════════════════

    "📦 Product Showcase": {
        "steps": 30, "cfg": 3.0, "sampler": "euler_ancestral_cfg_pp",
        "scheduler": "beta", "width": 1024, "height": 1024, "fps": 30, "length_s": 6.0,
        "nsfw_mode": "sfw", "audio_mode": "native",
        "prompt_template": "professional product showcase of {product_name}, "
                           "360 degree rotation, studio lighting, white background, "
                           "glossy surface, perfect shadows, commercial photography, "
                           "product advertisement, 8K render",
        "notes": "Quadrato 1:1 per Instagram / social. Carica immagine prodotto come frame 1.",
        "lora_suggestions": ["product_photography"],
    },

    "🍕 Food Photography": {
        "steps": 30, "cfg": 3.0, "sampler": "euler_ancestral_cfg_pp",
        "scheduler": "beta", "width": 1024, "height": 1024, "fps": 24, "length_s": 5.0,
        "nsfw_mode": "sfw", "audio_mode": "native",
        "prompt_template": "professional food photography of {dish_name}, "
                           "appetizing, steam rising, fresh ingredients, "
                           "rustic wooden table, warm lighting, shallow DOF, "
                           "Michelin star restaurant quality, 8K",
        "notes": "Per ristoranti, food blog, delivery app.",
        "lora_suggestions": [],
    },

    "👗 Fashion Editorial": {
        "steps": 35, "cfg": 3.0, "sampler": "dpmpp_2m_sde",
        "scheduler": "exponential", "width": 768, "height": 1280, "fps": 24, "length_s": 8.0,
        "nsfw_mode": "sfw", "audio_mode": "native",
        "prompt_template": "high fashion editorial photo of {subject} wearing {outfit}, "
                           "runway pose, Vogue magazine style, dramatic lighting, "
                           "editorial composition, luxury fashion, 8K",
        "notes": "Verticale 9:16 per Reels/TikTok. Combina con ControlNet Pose.",
        "lora_suggestions": ["fashion_model", "fashion_detail"],
    },

    "🚗 Auto Commerciale": {
        "steps": 40, "cfg": 3.5, "sampler": "dpmpp_2m_sde",
        "scheduler": "exponential", "width": 1280, "height": 544, "fps": 30, "length_s": 8.0,
        "nsfw_mode": "sfw", "audio_mode": "native",
        "prompt_template": "cinematic car advertisement, {car_model} driving on {location}, "
                           "dynamic camera angles, motion blur, dramatic lighting, "
                           "reflections on paint, premium automotive photography, 8K",
        "notes": "Anamorphic widescreen per feel cinematografico premium.",
        "lora_suggestions": [],
    },

    "💎 Gioielleria": {
        "steps": 40, "cfg": 3.5, "sampler": "dpmpp_2m_sde",
        "scheduler": "exponential", "width": 1024, "height": 1024, "fps": 30, "length_s": 5.0,
        "nsfw_mode": "sfw", "audio_mode": "native",
        "prompt_template": "luxury jewelry macro photography, {product}, sparkling gemstones, "
                           "reflective gold/silver surface, black velvet background, "
                           "studio lighting, ultra detailed, product advertisement",
        "notes": "Macro close-up per gioielli. Steps alti per massimo dettaglio.",
        "lora_suggestions": [],
    },

    # ══════════════════════════════════════════════════════════════════
    # AZIONE & SPORT
    # ══════════════════════════════════════════════════════════════════

    "⚡ Action Scene": {
        "steps": 35, "cfg": 3.5, "sampler": "dpmpp_2m_sde",
        "scheduler": "exponential", "width": 1280, "height": 544, "fps": 30, "length_s": 6.0,
        "nsfw_mode": "sfw", "audio_mode": "native",
        "prompt_template": "explosive action scene, {subject}, dynamic movement, motion blur, "
                           "debris flying, dramatic lighting, adrenaline, cinematic action movie, "
                           "slow motion effect, 8K HDR",
        "notes": "Anamorphic 2.39:1 per stile action movie. FPS alto per smoothness.",
        "lora_suggestions": ["action_detail"],
    },

    "🥋 Arti Marziali": {
        "steps": 35, "cfg": 3.0, "sampler": "dpmpp_2m_sde",
        "scheduler": "exponential", "width": 1280, "height": 704, "fps": 30, "length_s": 6.0,
        "nsfw_mode": "sfw", "audio_mode": "native",
        "prompt_template": "martial arts action shot, {subject} performing {technique}, "
                           "perfect form, dramatic pose, gym/dojo background, "
                           "motion blur on limbs, focus and power, 8K",
        "notes": "Usa ControlNet Pose per movimenti accurati. Carica video di riferimento.",
        "lora_suggestions": [],
    },

    "⚽ Sport All'aria Aperta": {
        "steps": 30, "cfg": 2.5, "sampler": "euler_ancestral_cfg_pp",
        "scheduler": "beta", "width": 1280, "height": 704, "fps": 30, "length_s": 6.0,
        "nsfw_mode": "sfw", "audio_mode": "native",
        "prompt_template": "dynamic sports photography, {athlete} playing {sport}, "
                           "stadium atmosphere, crowd cheering, peak action moment, "
                           "sports photography style, Canon 1DX quality, 8K",
        "notes": "Per highlight sportivi, recap partite, campagne sportswear.",
        "lora_suggestions": [],
    },

    "💃 Danza": {
        "steps": 32, "cfg": 2.8, "sampler": "euler_ancestral_cfg_pp",
        "scheduler": "beta", "width": 704, "height": 1280, "fps": 30, "length_s": 8.0,
        "nsfw_mode": "sfw", "audio_mode": "clone",
        "prompt_template": "graceful {dance_style} dancer, {subject}, dynamic movement, "
                           "stage lighting, flowing costume, perfect form, "
                           "expressive performance, concert hall",
        "notes": "Verticale 9:16 per social. Audio clone per sincronizzare con la musica.",
        "lora_suggestions": ["dancing_motion"],
    },

    # ══════════════════════════════════════════════════════════════════
    # FANTASY & SCI-FI
    # ══════════════════════════════════════════════════════════════════

    "🧙 Fantasy Epico": {
        "steps": 45, "cfg": 4.0, "sampler": "dpmpp_2m_sde",
        "scheduler": "exponential", "width": 1280, "height": 544, "fps": 24, "length_s": 10.0,
        "nsfw_mode": "sfw", "audio_mode": "native",
        "prompt_template": "epic fantasy scene, {subject}, magical atmosphere, glowing runes, "
                           "ancient castle, dragon in background, ethereal lighting, "
                           "concept art style, Frank Frazetta inspired, 8K",
        "notes": "Ideale per giochi, libri, trailer fantasy.",
        "lora_suggestions": ["fantasy_style"],
    },

    "🤖 Sci-Fi Cyberpunk": {
        "steps": 40, "cfg": 3.8, "sampler": "dpmpp_2m_sde",
        "scheduler": "exponential", "width": 1280, "height": 704, "fps": 24, "length_s": 8.0,
        "nsfw_mode": "sfw", "audio_mode": "native",
        "prompt_template": "cyberpunk cityscape, {subject}, neon lights, rain-slicked streets, "
                           "holographic displays, flying cars, dark atmosphere, "
                           "Blade Runner aesthetic, 8K HDR",
        "notes": "Stile Blade Runner / Ghost in the Shell.",
        "lora_suggestions": ["cyberpunk_style", "neon_detail"],
    },

    "🚀 Scena Spaziale": {
        "steps": 40, "cfg": 3.5, "sampler": "dpmpp_2m_sde",
        "scheduler": "exponential", "width": 1280, "height": 544, "fps": 24, "length_s": 10.0,
        "nsfw_mode": "sfw", "audio_mode": "native",
        "prompt_template": "epic space scene, {subject}, distant galaxies, nebulae, "
                           "starfield, planet surface, cosmic scale, "
                           "NASA/ESA quality, ultra detailed, 8K",
        "notes": "Per documentari spaziali, sci-fi, visualizzazioni astronomiche.",
        "lora_suggestions": [],
    },

    "🧟 Horror / Creatura": {
        "steps": 40, "cfg": 4.0, "sampler": "dpmpp_2m_sde",
        "scheduler": "exponential", "width": 1280, "height": 704, "fps": 24, "length_s": 8.0,
        "nsfw_mode": "sfw", "audio_mode": "native",
        "prompt_template": "terrifying horror scene, {subject}, dark atmosphere, "
                           "fog, abandoned location, eerie lighting, shadow creatures, "
                           "cinematic horror movie, jump scare composition",
        "notes": "Tono oscuro per horror, thriller, campagne Halloween.",
        "lora_suggestions": [],
    },

    "⚔️ Guerra Medievale": {
        "steps": 40, "cfg": 3.5, "sampler": "dpmpp_2m_sde",
        "scheduler": "exponential", "width": 1280, "height": 544, "fps": 24, "length_s": 8.0,
        "nsfw_mode": "sfw", "audio_mode": "native",
        "prompt_template": "epic medieval battle scene, {subject}, knights in armor, "
                           "swords clashing, castle siege, dramatic lighting, "
                           "Game of Thrones style, cinematic battle, 8K",
        "notes": "Cinemascope per massimo impatto epico.",
        "lora_suggestions": [],
    },

    # ══════════════════════════════════════════════════════════════════
    # ANIMAZIONE & ARTE
    # ══════════════════════════════════════════════════════════════════

    "🎨 Pittura Animata": {
        "steps": 50, "cfg": 4.5, "sampler": "dpmpp_2m_sde",
        "scheduler": "exponential", "width": 1024, "height": 768, "fps": 24, "length_s": 8.0,
        "nsfw_mode": "sfw", "audio_mode": "native",
        "prompt_template": "animated painting of {subject}, brushstrokes visible, "
                           "oil paint texture, impressionist style, rich colors, "
                           "museum quality, masterpiece, 8K",
        "notes": "Stile Van Gogh / Monet. Steps alti per dettaglio pennellate.",
        "lora_suggestions": ["painting_style", "oil_texture"],
    },

    "🖼️ Anime / Manga": {
        "steps": 30, "cfg": 2.5, "sampler": "euler_ancestral_cfg_pp",
        "scheduler": "beta", "width": 768, "height": 1024, "fps": 24, "length_s": 6.0,
        "nsfw_mode": "sfw", "audio_mode": "native",
        "prompt_template": "anime style, {subject}, manga art, clean line art, "
                           "cel shading, vibrant colors, Studio Ghibli quality, "
                           "beautiful character design, 4K",
        "notes": "Stile anime/manga. Combina con LoRA anime specifiche.",
        "lora_suggestions": ["anime_style"],
    },

    "🌈 Abstract / Visuals": {
        "steps": 45, "cfg": 4.5, "sampler": "dpmpp_2m_sde",
        "scheduler": "exponential", "width": 1024, "height": 1024, "fps": 30, "length_s": 8.0,
        "nsfw_mode": "sfw", "audio_mode": "native",
        "prompt_template": "abstract digital art, {subject}, fluid dynamics, "
                           "neon colors, geometric patterns, fractal elements, "
                           "motion graphics style, 8K",
        "notes": "Per VJ loops, music videos, screensaver art.",
        "lora_suggestions": [],
    },

    "🎮 Game Cinematics": {
        "steps": 40, "cfg": 3.5, "sampler": "dpmpp_2m_sde",
        "scheduler": "exponential", "width": 1280, "height": 720, "fps": 30, "length_s": 8.0,
        "nsfw_mode": "sfw", "audio_mode": "native",
        "prompt_template": "AAA video game cutscene, {character} in {game_world}, "
                           "photorealistic game graphics, Unreal Engine 5 quality, "
                           "dynamic lighting, cinematic composition, 8K",
        "notes": "Per trailer giochi, cinematic reveal, concept animation.",
        "lora_suggestions": [],
    },

    # ══════════════════════════════════════════════════════════════════
    # ARCHITETTURA & INTERNI
    # ══════════════════════════════════════════════════════════════════

    "🏠 Real Estate Tour": {
        "steps": 28, "cfg": 2.5, "sampler": "euler_ancestral_cfg_pp",
        "scheduler": "beta", "width": 1280, "height": 704, "fps": 30, "length_s": 8.0,
        "nsfw_mode": "sfw", "audio_mode": "native",
        "prompt_template": "luxury real estate interior, {room_type}, modern design, "
                           "natural lighting, clean minimalist style, high-end furniture, "
                           "architectural photography, 4K",
        "notes": "Per agenzie immobiliari, virtual tour, render architetturali.",
        "lora_suggestions": [],
    },

    "🏙️ Architecture Exterior": {
        "steps": 35, "cfg": 3.0, "sampler": "dpmpp_2m_sde",
        "scheduler": "exponential", "width": 1280, "height": 704, "fps": 24, "length_s": 8.0,
        "nsfw_mode": "sfw", "audio_mode": "native",
        "prompt_template": "architectural exterior render, {building_name}, {architectural_style}, "
                           "golden hour lighting, dramatic sky, "
                           "professional architectural photography, 8K",
        "notes": "Per render architetturali, presentazioni studio design.",
        "lora_suggestions": [],
    },

    "🏛️ Patrimonio Storico": {
        "steps": 35, "cfg": 3.0, "sampler": "dpmpp_2m_sde",
        "scheduler": "exponential", "width": 1280, "height": 704, "fps": 24, "length_s": 10.0,
        "nsfw_mode": "sfw", "audio_mode": "native",
        "prompt_template": "historic landmark, {monument_name}, dramatic sky, "
                           "golden light, tourists in background, UNESCO World Heritage feel, "
                           "travel documentary style, 8K",
        "notes": "Per documentari storici, turismo, archeologia.",
        "lora_suggestions": [],
    },

    # ══════════════════════════════════════════════════════════════════
    # DOCUMENTARIO & NEWS
    # ══════════════════════════════════════════════════════════════════

    "📺 News / Broadcast": {
        "steps": 25, "cfg": 2.5, "sampler": "euler_ancestral_cfg_pp",
        "scheduler": "beta", "width": 1280, "height": 704, "fps": 25, "length_s": 6.0,
        "nsfw_mode": "sfw", "audio_mode": "native",
        "prompt_template": "broadcast news studio, {anchor} at news desk, "
                           "professional TV lighting, lower thirds text overlay space, "
                           "blue background, CNN/BBC quality",
        "notes": "Standard broadcast 16:9 25fps PAL.",
        "lora_suggestions": [],
    },

    "🎙️ Podcast Video": {
        "steps": 25, "cfg": 2.5, "sampler": "euler_ancestral_cfg_pp",
        "scheduler": "beta", "width": 1280, "height": 704, "fps": 30, "length_s": 10.0,
        "nsfw_mode": "sfw", "audio_mode": "native",
        "prompt_template": "podcast recording studio, {host} and {guest} in conversation, "
                           "microphones on table, acoustic panels background, "
                           "warm studio lighting, relaxed professional atmosphere",
        "notes": "Per podcast, interviste, talk show digitali.",
        "lora_suggestions": [],
    },

    "🌍 Documentario Naturalistico": {
        "steps": 40, "cfg": 3.5, "sampler": "dpmpp_2m_sde",
        "scheduler": "exponential", "width": 1280, "height": 544, "fps": 24, "length_s": 10.0,
        "nsfw_mode": "sfw", "audio_mode": "native",
        "prompt_template": "wildlife documentary, {animal} in natural habitat, "
                           "{environment}, golden hour, telephoto lens compression, "
                           "BBC Planet Earth quality, 8K, nature photography",
        "notes": "Cinemascope per feel documentaristico professionale.",
        "lora_suggestions": [],
    },

    # ══════════════════════════════════════════════════════════════════
    # SPECIALI
    # ══════════════════════════════════════════════════════════════════

    "🔬 Macro Close-up": {
        "steps": 40, "cfg": 3.5, "sampler": "dpmpp_2m_sde",
        "scheduler": "exponential", "width": 1024, "height": 1024, "fps": 24, "length_s": 5.0,
        "nsfw_mode": "sfw", "audio_mode": "native",
        "prompt_template": "extreme macro photography of {subject}, microscopic detail, "
                           "shallow depth of field, diffused light, "
                           "ultra high detail, scientific beauty, 8K",
        "notes": "Per scienze, biologia, arte macro. Quadrato per massima versatilità.",
        "lora_suggestions": [],
    },

    "🚁 Drone / Aerial Shot": {
        "steps": 35, "cfg": 3.0, "sampler": "dpmpp_2m_sde",
        "scheduler": "exponential", "width": 1280, "height": 544, "fps": 30, "length_s": 8.0,
        "nsfw_mode": "sfw", "audio_mode": "native",
        "prompt_template": "aerial drone shot of {location}, bird's eye view, "
                           "vast landscape below, roads and buildings from above, "
                           "DJI Mavic quality, 4K aerial photography, {time_of_day}",
        "notes": "Anamorphic per feel cinematografico aereo. Ideale per intro locazione.",
        "lora_suggestions": [],
    },

    "🌊 Underwater / Subacqueo": {
        "steps": 38, "cfg": 3.2, "sampler": "dpmpp_2m_sde",
        "scheduler": "exponential", "width": 1280, "height": 704, "fps": 24, "length_s": 8.0,
        "nsfw_mode": "sfw", "audio_mode": "native",
        "prompt_template": "underwater photography, {subject}, coral reef, "
                           "tropical fish, sunrays through water, "
                           "National Geographic quality, ultra HD, teal and blue tones",
        "notes": "Per documentari marini, turismo subacqueo, pubblicità mare.",
        "lora_suggestions": [],
    },

    "🕯️ Atmosfera Noir": {
        "steps": 38, "cfg": 3.5, "sampler": "dpmpp_2m_sde",
        "scheduler": "exponential", "width": 1024, "height": 768, "fps": 24, "length_s": 8.0,
        "nsfw_mode": "sfw", "audio_mode": "native",
        "prompt_template": "film noir scene, {subject}, black and white, "
                           "hard shadows, venetian blind light pattern, "
                           "1940s atmosphere, detective aesthetic, cigarette smoke",
        "notes": "Stile noir classico. Perfetto per cortometraggi, trailer thriller.",
        "lora_suggestions": [],
    },

    "🌸 Stagionale / Festivo": {
        "steps": 28, "cfg": 2.5, "sampler": "euler_ancestral_cfg_pp",
        "scheduler": "beta", "width": 1280, "height": 704, "fps": 25, "length_s": 6.0,
        "nsfw_mode": "sfw", "audio_mode": "native",
        "prompt_template": "{season} atmosphere, {subject}, seasonal decorations, "
                           "warm festive lighting, joyful mood, "
                           "commercial advertising style, 4K",
        "notes": "Sostituisci {season} con Christmas/Spring/Halloween ecc.",
        "lora_suggestions": [],
    },

    "📱 Social Media Reel": {
        "steps": 20, "cfg": 2.0, "sampler": "euler_ancestral_cfg_pp",
        "scheduler": "beta", "width": 576, "height": 1024, "fps": 30, "length_s": 5.0,
        "nsfw_mode": "sfw", "audio_mode": "native",
        "prompt_template": "trendy social media content, {subject}, vibrant colors, "
                           "dynamic movement, attention-grabbing, Instagram/TikTok style, "
                           "modern aesthetic, engaging visual",
        "notes": "Ottimizzato per TikTok/Reels/Shorts. Steps bassi per velocità.",
        "lora_suggestions": [],
    },

    "⚡ Quick Preview": {
        "steps": 6, "cfg": 0.9, "sampler": "euler_ancestral_cfg_pp",
        "scheduler": "beta", "width": 480, "height": 832, "fps": 16, "length_s": 3.0,
        "nsfw_mode": "sfw", "audio_mode": "native",
        "prompt_template": "{subject}",
        "notes": "Preview velocissima per testare il prompt. Qualità bassa, generazione rapida.",
        "lora_suggestions": [],
    },

    "🎯 Massima Qualità": {
        "steps": 40, "cfg": 3.0, "sampler": "dpmpp_2m_sde",
        "scheduler": "exponential", "width": 1280, "height": 704, "fps": 24, "length_s": 8.0,
        "nsfw_mode": "sfw", "audio_mode": "native",
        "prompt_template": "{subject}, ultra high quality, masterpiece, 8K, award-winning",
        "notes": "Impostazioni massima qualità per output finali. Tempo lungo.",
        "lora_suggestions": [],
    },
}
