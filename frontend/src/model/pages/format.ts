/** 0.9863 -> "98.63%". Solo formatea: nunca se usa para comparar. */
export function percent(value: number): string {
  return `${(value * 100).toFixed(2)}%`;
}
