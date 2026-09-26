"""Avatar parlante: logica condivisa fra tab Avatar (GUI), REST API e worker LongCat/H3.

Moduli:
  timing      — durata audio → segmenti LongCat (stdlib-only, importato anche dal worker)
  audio_chain — voce (TTS/dialogo/clone) + musica/effetti + mix → wav pronto per l'avatar
  image_step  — immagine iniziale: da testo, da file, da personaggio (varianti) o da più riferimenti
  pipeline    — orchestrazione completa usata da /avatar/generate
"""
