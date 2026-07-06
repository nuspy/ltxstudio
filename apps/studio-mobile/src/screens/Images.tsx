import { FeatureGate } from '../components/FeatureGate';
import Stub from './_Stub';

export default function Images() {
  return (
    <FeatureGate cap="cap.gen_images">
      <Stub title="Immagini" phase="fase S2" />
    </FeatureGate>
  );
}
