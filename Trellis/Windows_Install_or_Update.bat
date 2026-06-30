@echo off
setlocal enabledelayedexpansion

echo WARNING. For this auto installer to work you need to have installed Python 3.11 and Git and FFmpeg
echo follow this tutorial : https://youtu.be/DrhUHnYfwC0

set UV_SKIP_WHEEL_FILENAME_CHECK=1
set UV_LINK_MODE=copy

git clone --recursive https://github.com/FurkanGozukara/Trellis_2_3D_Generator

cd Trellis_2_3D_Generator

git reset --hard

git pull

git clone https://github.com/FurkanGozukara/UniRig

cd UniRig

git reset --hard

git pull

cd ..

py --version >nul 2>&1
if "%ERRORLEVEL%" == "0" (
    echo Python launcher is available. Generating Python 3.11 VENV
    py -3.11 -m venv venv
) else (
    echo Python launcher is not available, generating VENV with default Python. Make sure that it is 3.11
    python -m venv venv
)

call .\venv\Scripts\activate.bat

python -m pip install --upgrade pip

pip install uv

cd ..

uv pip install -r requirements_trellis.txt

uv pip install --no-deps https://huggingface.co/MonsterMMORPG/Wan_GGUF/resolve/main/flex_gemm-1.0.0-cp311-cp311-win_amd64.whl
uv pip install --no-deps https://huggingface.co/MonsterMMORPG/Wan_GGUF/resolve/main/o_voxel-0.0.1-cp311-cp311-win_amd64.whl
uv pip install --no-deps https://huggingface.co/MonsterMMORPG/Wan_GGUF/resolve/main/spconv_cu131-2.3.8+torch2.9.1-cp311-cp311-win_amd64.whl

cd Trellis_2_3D_Generator
cd venv

git clone https://github.com/Luo-Yihao/FaithC.git
uv pip install --no-build-isolation --no-deps -e ./FaithC

cd ..
cd ..

cd Trellis_2_3D_Generator
python ensure_ultrashape_source.py
cd ..
REM deactivate

REM call .\Trellis_2_3D_Generator\venv_rig\Scripts\activate.bat

REM uv pip install torch_scatter torch_cluster -f https://data.pyg.org/whl/torch-2.8.0+cu129.html --no-cache-dir

REM Install bundled o_voxel CUDA extension (required by TRELLIS.2 pipelines)
REM python -m pip install .\Trellis_2_3D_Generator\o-voxel --no-build-isolation

Windows_Model_Download_Resume.bat

REM Pause to keep the command prompt open
pause
