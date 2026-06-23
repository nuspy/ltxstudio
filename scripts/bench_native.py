import time, json, sys, urllib.request
sys.path.insert(0, r"E:\ai\ltxstudio")
from core import native_driver as nd
def free():
    try:
        urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8189/free",
            data=b"{}", headers={"Content-Type":"application/json"}), timeout=30)
    except Exception as e:
        print("free err", e)
for i in range(45):
    if nd.is_up(): break
    time.sleep(2)
print("up:", nd.is_up(), flush=True)
MODELS=[
 ("FLUX dev",    r"E:\AI\Models\diffusion_models\Flux\FLUX_Dev.safetensors","flux",20,3.5),
 ("FLUX schnell",r"E:\AI\Models\Checkpoints\Flux\flux1-schnell-fp8.safetensors","flux",4,0.0),
 ("Z-Image",     r"E:\AI\Models\diffusion_models\Z-Image\Z_Image_Turbo_BF16.safetensors","zimage",8,1.0),
]
res={}
for name,model,arch,steps,g in MODELS:
    free(); time.sleep(2)
    base=dict(model=model,arch=arch,width=1024,height=1024,steps=steps,guidance=g)
    t=time.time(); r1=nd.run_and_wait("image",dict(base,prompt="a scenic landscape, detailed",seed=1),timeout=900)
    lg=round(time.time()-t,1)
    t=time.time(); r2=nd.run_and_wait("image",dict(base,prompt="a scenic landscape at dawn",seed=2),timeout=900)
    gen=round(time.time()-t,1)
    load=round(lg-gen,1)
    res[name]={"load_gen":lg,"gen":gen,"load":load,"steps":steps,"status":r1.get("status")}
    print(name, res[name], flush=True)
json.dump(res, open(r"E:\ai\ltxstudio\output\bench_native.json","w"), indent=2)
print("SAVED bench_native.json")
