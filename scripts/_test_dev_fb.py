import time, sys
sys.path.insert(0, r"E:\ai\ltxstudio")
from core import native_driver as nd
for i in range(45):
    if nd.is_up(): break
    time.sleep(2)
print("up:", nd.is_up(), flush=True)
M=r"E:\AI\Models\diffusion_models\Flux\FLUX_Dev.safetensors"
b=dict(model=M, arch="flux", width=1024, height=1024, steps=20, guidance=3.5)
t=time.time(); r1=nd.run_and_wait("image", dict(b, prompt="a medieval castle on a cliff", seed=31), timeout=700)
print("FLUXdev run1 (load+gen+FBcache):", r1.get("status"), round(time.time()-t),"s", flush=True)
t=time.time(); r2=nd.run_and_wait("image", dict(b, prompt="a medieval castle, sunset", seed=32), timeout=700)
print("FLUXdev run2 (gen cached):", r2.get("status"), round(time.time()-t),"s")
