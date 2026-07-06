// Download dei risultati: su Android salva in galleria (Filesystem) + Share; su web usa <a download>.
import { Capacitor } from '@capacitor/core';
import { Filesystem, Directory } from '@capacitor/filesystem';
import { Share } from '@capacitor/share';

async function blobToBase64(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const r = new FileReader();
    r.onloadend = () => resolve((r.result as string).split(',')[1] || '');
    r.onerror = reject;
    r.readAsDataURL(blob);
  });
}

export async function saveToDevice(url: string, filename: string): Promise<string> {
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`download ${resp.status}`);
  const blob = await resp.blob();

  if (Capacitor.isNativePlatform()) {
    const data = await blobToBase64(blob);
    const res = await Filesystem.writeFile({ path: filename, data, directory: Directory.Documents, recursive: true });
    try {
      await Share.share({ title: filename, url: res.uri });
    } catch {
      /* condivisione annullata */
    }
    return res.uri;
  }

  // web/PWA: scarica via link
  const objUrl = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = objUrl;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(objUrl);
  return filename;
}
