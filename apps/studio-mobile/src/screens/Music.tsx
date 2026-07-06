import { FeatureGate } from '../components/FeatureGate';
import Stub from './_Stub';

export default function Music() {
  return (
    <FeatureGate cap="cap.music_song">
      <Stub title="Musica & Videoclip" phase="fase S3" />
    </FeatureGate>
  );
}
