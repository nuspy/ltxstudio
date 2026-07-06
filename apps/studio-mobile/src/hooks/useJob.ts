import { useCallback, useState } from 'react';
import { runJob, type Job } from '../api';
import { rememberResult } from '../gallery';

// Sottomette un job e ne segue l'avanzamento (progress/message) fino al risultato.
export function useJob() {
  const [job, setJob] = useState<Job | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');

  const run = useCallback(async (path: string, body: unknown) => {
    setBusy(true);
    setError('');
    setJob(null);
    try {
      const j = await runJob(path, body, setJob);
      setJob(j);
      if (j.files) j.files.forEach((f) => rememberResult(f));
      return j;
    } catch (e: any) {
      setError(e?.message || String(e));
    } finally {
      setBusy(false);
    }
  }, []);

  return { job, busy, error, run };
}
