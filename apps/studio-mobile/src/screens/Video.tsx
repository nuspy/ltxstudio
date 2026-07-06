import { FeatureGate } from '../components/FeatureGate';
import Stub from './_Stub';

export default function Video() {
  return (
    <FeatureGate cap="cap.video">
      <Stub title="Video + ControlNet" phase="fase S3" />
    </FeatureGate>
  );
}
