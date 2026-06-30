cd Trellis_2_3D_Generator

call .\venv\Scripts\activate.bat

cd ..

set HF_HUB_ENABLE_HF_TRANSFER=1

REM Optional: set a Hugging Face token only if you need gated/private downloads.
REM set HUGGINGFACE_HUB_TOKEN=hf_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
python HF_model_downloader.py
python Trellis_2_3D_Generator\ensure_ultrashape_source.py

pause
