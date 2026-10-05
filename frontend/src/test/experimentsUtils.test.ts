import { describe, expect, it } from "vitest";

type MLflowRun = {
  info: { run_id: string; experiment_id: string };
  data: {
    metrics: Record<string, number>;
    tags: Record<string, string>;
  };
};

import { filterValidRuns, getMLflowLink, sortByMetric } from "../lib/experimentsUtils";

describe("Lógica de la Tabla de Experimentos (P3-12)", () => {
  const mockRuns: MLflowRun[] = [
    {
      info: { run_id: "abc", experiment_id: "0" },
      data: { metrics: { best_val_loss: 0.5 }, tags: { estado: "valida" } },
    },
    {
      info: { run_id: "def", experiment_id: "0" },
      data: { metrics: { best_val_loss: 0.2 }, tags: { estado: "invalida" } },
    },
    {
      info: { run_id: "ghi", experiment_id: "0" },
      data: { metrics: { best_val_loss: 0.8 }, tags: {} }, // Sin etiqueta
    },
  ];

  it("debe filtrar solo las corridas marcadas como válidas", () => {
    const validRuns = filterValidRuns(mockRuns);
    expect(validRuns).toHaveLength(1);
    expect(validRuns[0]?.info.run_id).toBe("abc");
  });

  it("debe ordenar las corridas por best_val_loss de menor a mayor", () => {
    const sorted = sortByMetric(mockRuns, "best_val_loss", "asc");
    expect(sorted[0]?.info.run_id).toBe("def");
    expect(sorted[1]?.info.run_id).toBe("abc");
    expect(sorted[2]?.info.run_id).toBe("ghi");
  });

  it("debe construir el enlace profundo exacto a la UI de MLflow", () => {
    const link = getMLflowLink("0", "abc");
    expect(link).toBe("http://localhost:5050/#/experiments/0/runs/abc");
  });
});
