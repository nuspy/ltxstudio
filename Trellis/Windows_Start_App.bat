@echo off

cd Trellis_2_3D_Generator

call .\venv\Scripts\activate.bat || exit /b

REM SET CUDA_VISIBLE_DEVICES=0  - this is used to set certain CUDA device visible only used
set PYTHONWARNINGS=ignore
set HF_HOME=models

set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8
set NVIDIA_TF32_OVERRIDE=1
set CUDA_MODULE_LOADING=LAZY

REM SET CUDA_VISIBLE_DEVICES=1
REM SET CUDA_VISIBLE_DEVICES=1

python app_premium.py



pause
