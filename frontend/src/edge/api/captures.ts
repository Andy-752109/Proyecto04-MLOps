import { mlApiRequest } from "@/model/api/client";
import { edgeCaptureListSchema } from "./contracts";

export function getEdgeCaptures(signal?: AbortSignal) {
  return mlApiRequest("/edge/captures?limit=100", edgeCaptureListSchema, {
    cache: "no-store",
    signal,
  });
}
