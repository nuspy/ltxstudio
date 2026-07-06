import { Preferences } from '@capacitor/preferences';

export type AppSettings = {
  serverUrl: string; // es. http://192.168.1.50:8200  oppure  https://fx.tadamoo.com
  token: string;     // bearer opzionale (server non-locale)
  theme: 'dark' | 'light';
  lang: string;
};

const KEY = 'ltx.settings';
export const DEFAULTS: AppSettings = {
  serverUrl: 'http://127.0.0.1:8200',
  token: '',
  theme: 'dark',
  lang: 'it',
};

export async function loadSettings(): Promise<AppSettings> {
  try {
    const { value } = await Preferences.get({ key: KEY });
    return value ? { ...DEFAULTS, ...JSON.parse(value) } : { ...DEFAULTS };
  } catch {
    return { ...DEFAULTS };
  }
}

export async function saveSettings(s: AppSettings): Promise<void> {
  await Preferences.set({ key: KEY, value: JSON.stringify(s) });
}
