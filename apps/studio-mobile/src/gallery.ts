import { Preferences } from '@capacitor/preferences';
import type { JobFile } from './api';

const KEY = 'ltx.gallery';

export async function loadResults(): Promise<JobFile[]> {
  try {
    const { value } = await Preferences.get({ key: KEY });
    return value ? JSON.parse(value) : [];
  } catch {
    return [];
  }
}

export async function rememberResult(f: JobFile): Promise<void> {
  const cur = await loadResults();
  const next = [f, ...cur.filter((x) => x.path !== f.path)].slice(0, 60);
  await Preferences.set({ key: KEY, value: JSON.stringify(next) });
}

export async function clearResults(): Promise<void> {
  await Preferences.remove({ key: KEY });
}
