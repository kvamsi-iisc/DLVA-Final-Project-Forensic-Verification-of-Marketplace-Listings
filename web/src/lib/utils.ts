import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export const pct = (x: number | null | undefined, digits = 0) =>
  x == null || Number.isNaN(x) ? "–" : `${(x * 100).toFixed(digits)}%`;

export function titleCase(s: string) {
  return s ? s[0].toUpperCase() + s.slice(1) : s;
}
