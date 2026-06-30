@echo off
REM Installer ACCELERATORI velocita'/memoria (Triton/FlashAttn/SageAttn/xFormers/DeepSpeed/ReActor),
REM distribution-aware: rileva OS/Python/torch/CUDA-arch e installa la strategia giusta per ciascuno.
REM Wheel arch-specific (flash_attn/sageattention) vanno messi in E:\AI\Tools\precompiled_wheels (docs\precompiled.pdf).
"E:\AI\Comfy_Nexus.bak\ComfyUI\venv\Scripts\python.exe" scripts\install_accelerators.py %*
echo.
echo (insightface/ReActor: usare  --venv "E:\AI\Tools\inference_server_venv\Scripts\python.exe" --only insightface,onnxruntime_gpu)
pause
