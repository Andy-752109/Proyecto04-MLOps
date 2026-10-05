import { useEffect, useState } from "react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { ML_API_BASE_URL } from "@/model/api/client";
export const ExperimentCurves = ({
  runId,
  metricKey = "val_loss",
}: {
  runId: string;
  metricKey?: string;
}) => {
  const [data, setData] = useState<{ step: number; value: number; timestamp: number }[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    fetch(`${ML_API_BASE_URL}/experiments/runs/${runId}/metrics/${metricKey}`)
      .then((res) => res.json())
      .then((history) => {
        setData(history);
        setLoading(false);
      })
      .catch((err) => {
        console.error("Error cargando métricas:", err);
        setLoading(false);
      });
  }, [runId, metricKey]);

  if (loading)
    return <div className="p-4 text-gray-500 text-sm">Cargando curvas de {metricKey}...</div>;
  if (!data.length)
    return (
      <div className="p-4 text-red-500 text-sm">No hay datos de {metricKey} para esta corrida.</div>
    );

  return (
    <div className="w-full h-64 mt-4 bg-white p-4 border rounded shadow-sm">
      <h3 className="text-sm font-semibold mb-2 text-gray-700">
        Historial de {metricKey} (Run: {runId.slice(0, 8)})
      </h3>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" vertical={false} />
          <XAxis dataKey="step" tick={{ fontSize: 12 }} tickFormatter={(value) => `Ep. ${value}`} />
          <YAxis
            domain={["auto", "auto"]}
            tick={{ fontSize: 12 }}
            tickFormatter={(value) => value.toFixed(3)}
          />
          <Tooltip
            formatter={(value: any) => {
              if (typeof value === "number") {
                return [value.toFixed(4), metricKey];
              }
              return [String(value ?? "N/A"), metricKey];
            }}
            labelFormatter={(label) => `Época ${label}`}
          />
          <Line
            type="monotone"
            dataKey="value"
            stroke="#2563eb"
            strokeWidth={2}
            dot={false}
            activeDot={{ r: 6 }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};
