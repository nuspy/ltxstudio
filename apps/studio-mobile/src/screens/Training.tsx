import { FeatureGate } from '../components/FeatureGate';
import Stub from './_Stub';

export default function Training() {
  return (
    <FeatureGate cap="cap.lora_image_train">
      <Stub title="Training LoRA" phase="fase S4" />
    </FeatureGate>
  );
}
