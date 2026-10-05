import { ExperimentsTable } from "@/components/experiments/ExperimentsTable";

export function ExperimentsPage() {
  return (
    <div className="p-6">
      <h1 className="text-2xl font-bold mb-4">Experimentos de MLflow en vivo</h1>
      <ExperimentsTable experimentId="1" />
    </div>
  );
}
