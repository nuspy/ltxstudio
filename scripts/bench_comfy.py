import time, json, urllib.request
P="http://127.0.0.1:8188"
def up():
    try: urllib.request.urlopen(P+"/system_stats",timeout=3); return True
    except: return False
def free():
    try:
        urllib.request.urlopen(urllib.request.Request(P+"/free",
          data=json.dumps({"unload_models":True,"free_memory":True}).encode(),
          headers={"Content-Type":"application/json"}),timeout=30)
    except Exception as e: print("free err",e)
def run(g):
    data=json.dumps({"prompt":g}).encode()
    r=json.load(urllib.request.urlopen(urllib.request.Request(P+"/prompt",data=data,headers={"Content-Type":"application/json"}),timeout=30))
    pid=r["prompt_id"]; t=time.time()
    for _ in range(300):
        time.sleep(2)
        h=json.load(urllib.request.urlopen(P+"/history/"+pid,timeout=10))
        if pid in h: return round(time.time()-t,1)
    return None
def flux_dev(prompt,seed):
    return {
 "1":{"class_type":"UNETLoader","inputs":{"unet_name":r"Flux\FLUX_Dev.safetensors","weight_dtype":"default"}},
 "2":{"class_type":"DualCLIPLoader","inputs":{"clip_name1":"t5xxl_fp16.safetensors","clip_name2":"long_clip_l_hi_dream.safetensors","type":"flux"}},
 "3":{"class_type":"VAELoader","inputs":{"vae_name":r"Flux\ae.safetensors"}},
 "4":{"class_type":"CLIPTextEncode","inputs":{"text":prompt,"clip":["2",0]}},
 "5":{"class_type":"FluxGuidance","inputs":{"conditioning":["4",0],"guidance":3.5}},
 "6":{"class_type":"CLIPTextEncode","inputs":{"text":"","clip":["2",0]}},
 "7":{"class_type":"EmptySD3LatentImage","inputs":{"width":1024,"height":1024,"batch_size":1}},
 "8":{"class_type":"KSampler","inputs":{"model":["1",0],"positive":["5",0],"negative":["6",0],"latent_image":["7",0],"seed":seed,"steps":20,"cfg":1.0,"sampler_name":"euler","scheduler":"simple","denoise":1.0}},
 "9":{"class_type":"VAEDecode","inputs":{"samples":["8",0],"vae":["3",0]}},
 "10":{"class_type":"SaveImage","inputs":{"images":["9",0],"filename_prefix":"bench_comfy"}}}
def schnell(prompt,seed):
    return {
 "1":{"class_type":"CheckpointLoaderSimple","inputs":{"ckpt_name":r"Flux\flux1-schnell-fp8.safetensors"}},
 "2":{"class_type":"CLIPTextEncode","inputs":{"text":prompt,"clip":["1",1]}},
 "4":{"class_type":"CLIPTextEncode","inputs":{"text":"","clip":["1",1]}},
 "5":{"class_type":"EmptySD3LatentImage","inputs":{"width":1024,"height":1024,"batch_size":1}},
 "6":{"class_type":"KSampler","inputs":{"model":["1",0],"positive":["2",0],"negative":["4",0],"latent_image":["5",0],"seed":seed,"steps":4,"cfg":1.0,"sampler_name":"euler","scheduler":"simple","denoise":1.0}},
 "7":{"class_type":"VAEDecode","inputs":{"samples":["6",0],"vae":["1",2]}},
 "8":{"class_type":"SaveImage","inputs":{"images":["7",0],"filename_prefix":"bench_comfy_s"}}}
for i in range(90):
    if up(): break
    time.sleep(2)
print("comfy up:", up(), flush=True)
res={}
for name,fn in [("FLUX dev",flux_dev),("FLUX schnell",schnell)]:
    free(); time.sleep(3)
    lg=run(fn("a scenic landscape, detailed",1))
    gen=run(fn("a scenic landscape at dawn",2))
    res[name]={"load_gen":lg,"gen":gen,"load":round((lg-gen),1) if lg and gen else None}
    print(name,res[name],flush=True)
json.dump(res, open(r"E:\ai\ltxstudio\output\bench_comfy.json","w"), indent=2)
print("SAVED bench_comfy.json")
