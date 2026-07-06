// Client REST verso il server LTX Studio (:8200). Base-url + token dai Settings.
import { loadSettings } from './settings';
import type { Snapshot } from './features';

export type JobFile = { path: string; name: string; download_url: string };
export type Job = {
  job_id: string;
  status: 'pending' | 'running' | 'success' | 'error';
  progress?: number;
  message?: string;
  queue_position?: number;
  files?: JobFile[];
  error?: string;
};

let base = 'http://127.0.0.1:8200';
let token = '';

export async function initApi(): Promise<void> {
  const s = await loadSettings();
  base = s.serverUrl.replace(/\/+$/, '');
  token = s.token || '';
}

export function setApi(url: string, tok: string): void {
  base = url.replace(/\/+$/, '');
  token = tok || '';
}

export function baseUrl(): string {
  return base;
}

function authHeaders(json = true): Record<string, string> {
  const h: Record<string, string> = {};
  if (json) h['Content-Type'] = 'application/json';
  if (token) h['Authorization'] = `Bearer ${token}`;
  return h;
}

export async function apiGet<T = any>(path: string): Promise<T> {
  const r = await fetch(base + path, { headers: authHeaders(false) });
  if (!r.ok) throw new Error(`GET ${path} → ${r.status} ${await r.text()}`);
  return r.json();
}

export async function apiPost<T = any>(path: string, body: unknown): Promise<T> {
  const r = await fetch(base + path, { method: 'POST', headers: authHeaders(), body: JSON.stringify(body) });
  if (!r.ok) throw new Error(`POST ${path} → ${r.status} ${await r.text()}`);
  return r.json();
}

export async function uploadFile(file: Blob, name: string): Promise<JobFile> {
  const fd = new FormData();
  fd.append('file', file, name);
  const h: Record<string, string> = {};
  if (token) h['Authorization'] = `Bearer ${token}`;
  const r = await fetch(base + '/files/upload', { method: 'POST', headers: h, body: fd });
  if (!r.ok) throw new Error(`upload → ${r.status}`);
  return r.json();
}

export async function health(): Promise<{ api: string; native_backend_up: boolean }> {
  return apiGet('/health');
}

export async function fetchFeatures(): Promise<Snapshot> {
  return apiGet('/features');
}

// polling di un job async fino a success|error
export async function pollJob(
  jobId: string,
  onUpdate: (j: Job) => void,
  intervalMs = 1500,
): Promise<Job> {
  return new Promise((resolve, reject) => {
    const tick = async () => {
      try {
        const j = await apiGet<Job>(`/jobs/${jobId}`);
        onUpdate(j);
        if (j.status === 'success') resolve(j);
        else if (j.status === 'error') reject(new Error(j.error || 'job error'));
        else setTimeout(tick, intervalMs);
      } catch (e) {
        reject(e);
      }
    };
    tick();
  });
}

// helper: sottometti un POST che ritorna {job_id} e attendi il risultato
export async function runJob(path: string, body: unknown, onUpdate: (j: Job) => void): Promise<Job> {
  const { job_id } = await apiPost<{ job_id: string }>(path, body);
  return pollJob(job_id, onUpdate);
}
