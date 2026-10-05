import { mlApiRequest } from "./client";
import type { CreateTrainingJobRequest } from "./contracts";
import { trainingJobListSchema, trainingJobSchema } from "./contracts";

export async function getTrainingJobs(signal?: AbortSignal) {
  return mlApiRequest("/training/jobs", trainingJobListSchema, { signal });
}

export async function createTrainingJob(request: CreateTrainingJobRequest) {
  return mlApiRequest("/training/jobs", trainingJobSchema, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
  });
}

// Revisión de Uriel sobre P3-09 ("no hay forma de cancelar"): `queued` se
// cancela directo; `running` solo deja una marca que trainer-worker revisa
// entre mensajes de progreso (ver app/ml_api/repository.py).
export async function cancelTrainingJob(id: string) {
  return mlApiRequest(`/training/jobs/${encodeURIComponent(id)}/cancel`, trainingJobSchema, {
    method: "POST",
  });
}
