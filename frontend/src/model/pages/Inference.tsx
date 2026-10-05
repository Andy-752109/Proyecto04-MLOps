import { type FormEvent, useState } from "react";
import { Dropzone } from "@/components/upload/Dropzone";
import { ApiError } from "@/lib/api/client";
import { createInferenceSubmission, uploadImage } from "@/lib/api/images";
import { PageHeader } from "@/pipeline/components/PageHeader";
import { MlApiError } from "../api/client";
import type { PredictionResponse } from "../api/contracts";
import { predictFromCrop, predictFromImage, validatePredictImageFile } from "../api/inference";

type Mode = "upload" | "crop";

type PredictState =
  | { status: "idle" }
  | { status: "loading" }
  | { status: "error"; message: string }
  | { status: "success"; result: PredictionResponse };

type QueueState =
  | { status: "idle" }
  | { status: "loading" }
  | { status: "error"; message: string }
  | { status: "success"; imageId: number };

// Mismo criterio que `Provenance` en Training.tsx: las clases se muestran
// como vienen del contrato ("cat"/"dog"), sin traducir -- este portal no
// mezcla idiomas dentro del mismo dato en ningún otro lado.
function ProbabilityBars({ probabilities }: { probabilities: Record<string, number> }) {
  const entries = Object.entries(probabilities).sort((a, b) => b[1] - a[1]);
  return (
    <div className="flex flex-col gap-2">
      {entries.map(([label, probability]) => (
        <div key={label} className="flex flex-col gap-1">
          <div className="flex justify-between text-xs text-ink-muted">
            <span>{label}</span>
            <span>{Math.round(probability * 100)}%</span>
          </div>
          <div className="h-2 rounded-full bg-sidebar">
            <div
              className="h-2 rounded-full bg-accent-lilac"
              style={{ width: `${Math.round(probability * 100)}%` }}
            />
          </div>
        </div>
      ))}
    </div>
  );
}

/**
 * "Enviar a cola de anotación" (P3-16, #24, checklist punto 5): reutiliza
 * `uploadImage` (image-upload.service.ts, backend) para crear la imagen en
 * `pending`, y adjunta la sugerencia del modelo con un endpoint nuevo. Solo
 * tiene sentido para el flujo de imagen nueva -- un recorte existente ya
 * tiene una imagen en el sistema, no hay nada que subir.
 */
function SendToQueueButton({ file, result }: { file: File; result: PredictionResponse }) {
  const [state, setState] = useState<QueueState>({ status: "idle" });

  async function handleClick() {
    setState({ status: "loading" });
    try {
      const uploaded = await uploadImage(file);
      await createInferenceSubmission(uploaded.id, {
        predictedLabel: result.predicted_label,
        probabilities: result.probabilities,
        modelVersion: result.model_version,
        checkpointSha256: result.checkpoint_sha256,
      });
      setState({ status: "success", imageId: uploaded.id });
    } catch (error) {
      setState({
        status: "error",
        message: error instanceof ApiError ? error.message : "No se pudo enviar a la cola.",
      });
    }
  }

  if (state.status === "success") {
    return <p className="text-sm text-status-done">Enviada a la cola (imagen #{state.imageId}).</p>;
  }

  return (
    <div className="flex flex-col gap-1">
      <button
        type="button"
        onClick={handleClick}
        disabled={state.status === "loading"}
        className="self-start rounded-full border border-border px-4 py-2 text-sm disabled:opacity-40"
      >
        {state.status === "loading" ? "Enviando…" : "Enviar a cola de anotación"}
      </button>
      {state.status === "error" && <p className="text-sm text-status-pending">{state.message}</p>}
    </div>
  );
}

function PredictionResult({ result, file }: { result: PredictionResponse; file?: File }) {
  return (
    <div className="flex flex-col gap-3 rounded-2xl border border-border bg-surface p-4 text-sm">
      <p>
        Predicción: <span className="font-medium text-ink">{result.predicted_label}</span>
      </p>
      <ProbabilityBars probabilities={result.probabilities} />
      <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 text-xs text-ink-muted">
        <dt>versión del modelo</dt>
        <dd className="font-mono">{result.model_version}</dd>
        <dt>checkpoint SHA-256</dt>
        <dd className="break-all font-mono">{result.checkpoint_sha256}</dd>
      </dl>
      {file && <SendToQueueButton file={file} result={result} />}
    </div>
  );
}

function UploadForm() {
  const [state, setState] = useState<PredictState>({ status: "idle" });
  const [file, setFile] = useState<File | null>(null);

  async function handleFiles(files: FileList | File[]) {
    const selected = Array.from(files)[0];
    if (!selected) return;
    setFile(selected);
    const validationError = validatePredictImageFile(selected);
    if (validationError) {
      setState({ status: "error", message: validationError });
      return;
    }
    setState({ status: "loading" });
    try {
      const result = await predictFromImage(selected);
      setState({ status: "success", result });
    } catch (error) {
      setState({
        status: "error",
        message: error instanceof MlApiError ? error.message : "No se pudo clasificar la imagen.",
      });
    }
  }

  return (
    <div className="flex flex-col gap-3">
      <Dropzone onFiles={handleFiles} />
      {state.status === "loading" && <p className="text-sm text-ink-muted">Clasificando…</p>}
      {state.status === "error" && <p className="text-sm text-status-pending">{state.message}</p>}
      {state.status === "success" && (
        <PredictionResult result={state.result} file={file ?? undefined} />
      )}
    </div>
  );
}

function CropForm() {
  const [imageId, setImageId] = useState("");
  const [annotationId, setAnnotationId] = useState("");
  const [state, setState] = useState<PredictState>({ status: "idle" });

  async function submit(event: FormEvent) {
    event.preventDefault();
    const parsedImageId = Number(imageId);
    const parsedAnnotationId = Number(annotationId);
    if (!Number.isInteger(parsedImageId) || !Number.isInteger(parsedAnnotationId)) {
      setState({ status: "error", message: "image_id y annotation_id deben ser enteros." });
      return;
    }
    setState({ status: "loading" });
    try {
      const result = await predictFromCrop(parsedImageId, parsedAnnotationId);
      setState({ status: "success", result });
    } catch (error) {
      setState({
        status: "error",
        message: error instanceof MlApiError ? error.message : "No se pudo clasificar el recorte.",
      });
    }
  }

  return (
    <form onSubmit={submit} className="flex flex-col gap-3">
      <div className="flex gap-3">
        <label htmlFor="image_id" className="flex flex-col gap-1 text-sm">
          image_id
          <input
            id="image_id"
            className="rounded border border-border bg-surface p-2"
            value={imageId}
            onChange={(event) => setImageId(event.target.value)}
          />
        </label>
        <label htmlFor="annotation_id" className="flex flex-col gap-1 text-sm">
          annotation_id
          <input
            id="annotation_id"
            className="rounded border border-border bg-surface p-2"
            value={annotationId}
            onChange={(event) => setAnnotationId(event.target.value)}
          />
        </label>
      </div>
      <button
        type="submit"
        className="self-start rounded-full bg-ink px-4 py-2 text-sm font-medium text-white"
      >
        Predecir
      </button>
      {state.status === "loading" && <p className="text-sm text-ink-muted">Clasificando…</p>}
      {state.status === "error" && <p className="text-sm text-status-pending">{state.message}</p>}
      {state.status === "success" && <PredictionResult result={state.result} />}
    </form>
  );
}

export function InferencePage() {
  const [mode, setMode] = useState<Mode>("upload");

  return (
    <main className="flex-1 px-6 py-6 lg:px-10 lg:py-8">
      <div className="mx-auto flex max-w-3xl flex-col gap-6">
        <PageHeader title="Inference" subtitle="Clasificar una imagen con el modelo publicado." />

        <div className="flex gap-2" role="tablist">
          <button
            type="button"
            role="tab"
            aria-selected={mode === "upload"}
            onClick={() => setMode("upload")}
            className={`rounded-full px-4 py-2 text-sm ${
              mode === "upload" ? "bg-ink text-white" : "border border-border"
            }`}
          >
            Imagen nueva
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={mode === "crop"}
            onClick={() => setMode("crop")}
            className={`rounded-full px-4 py-2 text-sm ${
              mode === "crop" ? "bg-ink text-white" : "border border-border"
            }`}
          >
            Recorte existente
          </button>
        </div>

        {mode === "upload" ? <UploadForm /> : <CropForm />}
      </div>
    </main>
  );
}
