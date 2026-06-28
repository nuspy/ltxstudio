import time, json, urllib.request, sys
sys.path.insert(0, r"E:\ai\ltxstudio")
from core.extend_pipeline import ExtendPipeline
P="http://127.0.0.1:8188"
g=ExtendPipeline().build("extend_src.mp4", "a cat walking in a sunny garden, continue the motion",
           width=512, height=512, fps=24, add_seconds=2.0, seg_seconds=2.0,
           steps=8, cfg=1.0, seed=2, unet=r"LTX\ltx-2.3-22b-dev-fp8.safetensors",
           clip1="gemma_3_12B_it.safetensors")
print("nodi:", len(g), flush=True)
data=json.dumps({"prompt":g}).encode()
try:
    r=json.load(urllib.request.urlopen(urllib.request.Request(P+"/prompt",data=data,headers={"Content-Type":"application/json"}),timeout=40)); pid=r["prompt_id"]; print("submitted",pid,flush=True)
except urllib.error.HTTPError as e:
    print("HTTP",e.code,e.read().decode()[:400]); raise SystemExit
t=time.time()
for _ in range(600):
    time.sleep(3); h=json.load(urllib.request.urlopen(P+"/history/"+pid,timeout=10))
    if pid in h:
        st=h[pid]["status"].get("status_str"); outs=h[pid].get("outputs",{})
        vids=[v["filename"] for n in outs.values() for v in (n.get("gifs",[])+n.get("videos",[]))]
        print(f"[extend] {st} in {round(time.time()-t)}s -> video={vids}")
        if st=="error":
            for m in h[pid]["status"].get("messages",[]):
                if m[0]=="execution_error": print("ERR:", m[1].get("node_type"), str(m[1].get("exception_message"))[:220])
        break
