/* LTX Studio — cattura del "golden prompt" (formato API) dal frontend ComfyUI.
 *
 * COME USARLO:
 * 1. Avvia ComfyUI (run_nvidia.bat) e apri http://127.0.0.1:8188 nel browser.
 * 2. Carica il workflow Pluto_AllInOne_v2.json.
 * 3. Configura i moduli che vuoi includere nel template (per il SUPERSET: attiva
 *    i Fast Groups Bypasser dei moduli che ti servono; per un primo test basta
 *    la modalità Text-to-Video base).
 * 4. Apri DevTools (F12) -> Console e incolla TUTTO questo file, premi Invio.
 * 5. Parte il download di "pluto_<modalita>.api.json": salvalo in
 *    E:\AI\ltxstudio\templates\
 *
 * graphToPrompt() risolve Set/Get virtuali, subgraph e bypass nel formato API
 * piatto {node_id: {class_type, inputs}} accettato da POST /prompt.
 */
(async () => {
  if (typeof app === "undefined" || !app.graphToPrompt) {
    alert("app.graphToPrompt non disponibile: assicurati di essere nella pagina di ComfyUI.");
    return;
  }
  const res = await app.graphToPrompt();
  const apiPrompt = res.output;          // formato API (prompt)
  const uiGraph  = res.workflow;         // formato UI (per riferimento)
  const bundle = { api: apiPrompt, ui_node_count: Object.keys(apiPrompt).length };
  const json = JSON.stringify(apiPrompt, null, 2);
  const blob = new Blob([json], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "pluto_capture.api.json";
  document.body.appendChild(a);
  a.click();
  a.remove();
  console.log("[LTX Studio] golden prompt catturato:",
              Object.keys(apiPrompt).length, "nodi. File scaricato: pluto_capture.api.json");
})();
