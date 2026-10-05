import { useEffect, useState } from "react";
import { PageHeader } from "@/pipeline/components/PageHeader";
import { getPending } from "../api/pending";

type PendingPath = "/experiments" | "/evaluation" | "/models" | "/inference";

/**
 * Las 4 áreas de "Modelo" sin datos reales todavía (P3-12..P3-16). Muestra
 * "Pendiente" siempre -- no depende de que el fetch a ml-api tenga éxito --
 * y el motivo real de ml-api como detalle, cuando carga.
 */
export function PendingPage({
  title,
  subtitle,
  path,
}: Readonly<{ title: string; subtitle: string; path: PendingPath }>) {
  const [detail, setDetail] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    getPending(path)
      .then((result) => {
        if (active) setDetail(result.message);
      })
      .catch(() => {
        // Se queda sin detalle; el texto "Pendiente" ya cuenta la historia completa.
      });
    return () => {
      active = false;
    };
  }, [path]);

  return (
    <main className="flex-1 px-6 py-6 lg:px-10 lg:py-8">
      <div className="mx-auto flex max-w-6xl flex-col gap-6">
        <PageHeader title={title} subtitle={subtitle} />
        <p className="text-sm text-ink-muted">Pendiente{detail ? `: ${detail}` : "."}</p>
      </div>
    </main>
  );
}
