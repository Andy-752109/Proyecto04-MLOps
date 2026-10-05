import { useEffect, useMemo, useState } from "react";
import { z } from "zod";
import { mlApiRequest } from "@/model/api/client";
import {
  filterValidRuns,
  getMLflowLink,
  type MLflowRun,
  sortByMetric,
} from "../../lib/experimentsUtils";
import { ExperimentCurves } from "./ExperimentCurves";
export const ExperimentsTable = ({ experimentId = "0" }) => {
  const [runs, setRuns] = useState<MLflowRun[]>([]);
  const [onlyValid, setOnlyValid] = useState(false);
  const [sortOrder, setSortOrder] = useState<"asc" | "desc">("asc");
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);

  useEffect(() => {
    mlApiRequest(`/experiments/${experimentId}/runs`, z.array(z.any()))
      .then((data) => {
        setRuns(data);
      })
      .catch((err) => {
        console.error("Error al cargar corridas:", err);
        setRuns([]);
      });
  }, [experimentId]);

  const displayedRuns = useMemo(() => {
    let processed = runs;
    if (onlyValid) {
      processed = filterValidRuns(processed);
    }
    return sortByMetric(processed, "best_val_loss", sortOrder);
  }, [runs, onlyValid, sortOrder]);

  const toggleValidStatus = async (runId: string, currentStatus: string | undefined) => {
    const newStatus = currentStatus === "valida" ? "invalida" : "valida";

    try {
      await mlApiRequest(`/experiments/runs/${runId}/tags`, z.any(), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ key: "estado", value: newStatus }),
      });

      setRuns((prevRuns) =>
        prevRuns.map((run) =>
          run.info.run_id === runId
            ? { ...run, data: { ...run.data, tags: { ...run.data.tags, estado: newStatus } } }
            : run
        )
      );
    } catch (err) {
      console.error("Error actualizando el estado:", err);
    }
  };

  return (
    <div className="p-4">
      <div className="mb-4 flex gap-4">
        <button
          type="button"
          onClick={() => setOnlyValid(!onlyValid)}
          className="px-4 py-2 bg-blue-500 text-white rounded"
        >
          {onlyValid ? "Mostrar Todas" : "Solo Válidas"}
        </button>
        <button
          type="button"
          onClick={() => setSortOrder(sortOrder === "asc" ? "desc" : "asc")}
          className="px-4 py-2 border rounded"
        >
          Orden Loss: {sortOrder.toUpperCase()}
        </button>
      </div>

      <table className="min-w-full border-collapse border">
        <thead>
          <tr className="bg-gray-100">
            <th className="border p-2">Run ID</th>
            <th className="border p-2">best_val_loss</th>
            <th className="border p-2">Estado</th>
            <th className="border p-2">MLflow UI</th>
          </tr>
        </thead>
        <tbody>
          {displayedRuns.map((run) => (
            <tr key={run.info.run_id} className="text-center">
              <td className="border p-2 font-mono text-sm">
                <button
                  type="button"
                  onClick={() => setSelectedRunId(run.info.run_id)}
                  className="text-blue-600 hover:underline font-bold"
                >
                  {run.info.run_id.slice(0, 8)}
                </button>
              </td>
              <td className="border p-2">{run.data.metrics?.best_val_loss?.toFixed(4) ?? "N/A"}</td>
              <td className="border p-2">
                <button
                  type="button"
                  onClick={() => toggleValidStatus(run.info.run_id, run.data.tags?.estado)}
                  className={`px-3 py-1 rounded text-sm ${
                    run.data.tags?.estado === "valida"
                      ? "bg-green-200 text-green-800"
                      : "bg-red-200 text-red-800"
                  }`}
                >
                  {run.data.tags?.estado === "valida" ? "Válida" : "Inválida"}
                </button>
              </td>
              <td className="border p-2">
                <a
                  href={getMLflowLink(experimentId, run.info.run_id)}
                  target="_blank"
                  rel="noreferrer"
                  className="text-blue-600 underline text-sm"
                >
                  Ver gráficas nativas
                </a>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      {selectedRunId && (
        <div className="mt-8 grid grid-cols-2 gap-4">
          <ExperimentCurves runId={selectedRunId} metricKey="val_loss" />
          <ExperimentCurves runId={selectedRunId} metricKey="train_loss" />
        </div>
      )}
    </div>
  );
};
