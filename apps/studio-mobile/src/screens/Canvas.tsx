import { FeatureGate } from '../components/FeatureGate';
import Stub from './_Stub';

export default function Canvas() {
  return (
    <FeatureGate cap="cap.inpaint">
      <Stub title="Canvas & Mask" phase="fase S2" />
    </FeatureGate>
  );
}
