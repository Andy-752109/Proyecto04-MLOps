import { z } from "zod";
import { useReportFetch } from "@/pipeline/useReportFetch";

/**
 * `reports/manifests/<release>/manifest_meta.json` y `counts.json` (P3-06,
 * ver `app/manifest/README.md`) -- JSON estático servido por nginx, igual
 * que `reports/versions.json` (`@/pipeline/dataSource`). "Procedencia" del
 * checklist de P3-09: hash del manifiesto, `manifest_id`, SHA-256 y
 * conteos por clase y partición.
 */

export const manifestMetaSchema = z
  .object({
    manifest_id: z.string().min(1),
    manifest_sha256: z.string().min(1),
    release: z.object({ name: z.string().min(1) }),
    seed: z.number().int(),
    classes: z.record(z.string(), z.string()),
    rows: z.number().int().nonnegative(),
  })
  .passthrough();

export type ManifestMeta = z.infer<typeof manifestMetaSchema>;

const classCountsSchema = z.record(
  z.string(),
  z.object({ crops: z.number().int().nonnegative(), originals: z.number().int().nonnegative() })
);

const splitCountsSchema = z
  .object({
    target_fraction: z.number().finite(),
    crops: z.number().int().nonnegative(),
    originals: z.number().int().nonnegative(),
    crops_fraction: z.number().finite(),
    deviation_pp: z.number().finite(),
    classes: classCountsSchema,
  })
  .passthrough();

export const manifestCountsSchema = z
  .object({
    release: z.string().min(1),
    totals: z.object({
      crops: z.number().int().nonnegative(),
      originals: z.number().int().nonnegative(),
      classes: classCountsSchema,
    }),
    splits: z.record(z.string(), splitCountsSchema),
  })
  .passthrough();

export type ManifestCounts = z.infer<typeof manifestCountsSchema>;

export function useManifestMeta(release: string | null) {
  return useReportFetch(
    release === null ? null : `/reports/manifests/${release}/manifest_meta.json`,
    manifestMetaSchema
  );
}

export function useManifestCounts(release: string | null) {
  return useReportFetch(
    release === null ? null : `/reports/manifests/${release}/counts.json`,
    manifestCountsSchema
  );
}
