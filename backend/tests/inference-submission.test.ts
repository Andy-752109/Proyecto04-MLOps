import { describe, expect, it } from 'vitest';

import { validateInferenceSubmission } from '../src/logic/inference-submission.validation.js';

/**
 * P3-16 (#24): validación de la sugerencia que se guarda al "enviar a cola
 * de anotación" -- espejo de `PredictionResponse` (ml_api/contracts.py).
 */
describe('P3-16 - validación de inference_submissions', () => {
  const valid = {
    predictedLabel: 'cat',
    probabilities: { cat: 0.9, dog: 0.1 },
    modelVersion: '1.0.0',
    checkpointSha256: 'a'.repeat(64),
  };

  it('acepta una sugerencia válida', () => {
    expect(validateInferenceSubmission(valid).success).toBe(true);
  });

  it('rechaza una probabilidad fuera de [0,1]', () => {
    const result = validateInferenceSubmission({
      ...valid,
      probabilities: { cat: 1.5, dog: 0.1 },
    });
    expect(result.success).toBe(false);
  });

  it('rechaza un checkpoint_sha256 que no sea de 64 caracteres', () => {
    const result = validateInferenceSubmission({ ...valid, checkpointSha256: 'abc' });
    expect(result.success).toBe(false);
  });

  it('rechaza predictedLabel vacío', () => {
    const result = validateInferenceSubmission({ ...valid, predictedLabel: '' });
    expect(result.success).toBe(false);
  });

  it('rechaza campos faltantes', () => {
    const { modelVersion, ...incomplete } = valid;
    expect(validateInferenceSubmission(incomplete).success).toBe(false);
  });
});
