// Gating feature-matrix lato-app — EDITION-AWARE (stesse attivazioni del desktop).
// `edition.json` è bakeato per edizione dal builder feature_matrix (dev = vuoto → fail-open, tutto abilitato,
// come la GUI desktop). A runtime si può fondere con lo snapshot del server (/features). Priorità: baked → server → fail-open.
import editionData from './edition.json';

export type CapInfo = { enabled: boolean; i18n_key?: string; upsell?: Upsell | null };
export type Upsell = { tier?: string; tier_name?: string; price?: string; url?: string } | null;
export type Snapshot = { edition: string; allows_adult: boolean; features: Record<string, CapInfo> };

const baked = editionData as Snapshot;
let serverSnap: Snapshot | null = null;

export function setServerSnapshot(s: Snapshot | null) {
  serverSnap = s;
}

export function edition(): string {
  return (baked.edition && baked.edition !== 'dev' ? baked.edition : serverSnap?.edition) || baked.edition || 'dev';
}

export function isEnabled(cap: string): boolean {
  const b = baked.features?.[cap];
  if (b && typeof b.enabled === 'boolean') return b.enabled; // APK dell'edizione = autoritativo
  const s = serverSnap?.features?.[cap];
  if (s && typeof s.enabled === 'boolean') return s.enabled;
  return true; // dev / capability assente → fail-open (come desktop)
}

export function capInfo(cap: string): CapInfo | undefined {
  return baked.features?.[cap] ?? serverSnap?.features?.[cap];
}

export function upsell(cap: string): Upsell {
  return capInfo(cap)?.upsell ?? null;
}

export function allowsAdult(): boolean {
  if (baked.edition && baked.edition !== 'dev' && typeof baked.allows_adult === 'boolean') return baked.allows_adult;
  return serverSnap?.allows_adult ?? true;
}
