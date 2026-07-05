[app]
title = ATF Detector
package.name = atfdetector
package.domain = com.aitoolsforge
source.dir = .
source.include_exts = py,png,jpg,kv,ttf
version = 1.0

# Solo dipendenze pure/leggere → build p4a affidabile. NIENTE opencv/scipy: il rilevamento è in numpy+PIL.
requirements = python3,kivy,numpy,pillow,plyer,android

orientation = portrait
fullscreen = 0

# Permessi: fotocamera + lettura foto (READ_MEDIA_IMAGES per Android 13+, READ_EXTERNAL_STORAGE per i più vecchi)
android.permissions = CAMERA, READ_MEDIA_IMAGES, READ_EXTERNAL_STORAGE
android.api = 34
android.minapi = 24
android.archs = arm64-v8a, armeabi-v7a
android.allow_backup = 1

# icona/presentazione (opzionali — aggiungi i file se vuoi)
# icon.filename = %(source.dir)s/icon.png
# presplash.filename = %(source.dir)s/presplash.png

[buildozer]
log_level = 2
warn_on_root = 1
