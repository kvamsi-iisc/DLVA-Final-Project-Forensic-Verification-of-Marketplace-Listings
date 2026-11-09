export type Score = { p_fake: number; threshold: number | null; flag: boolean };
export type Authenticity = { probe: Score; spectrum?: Score; fused?: Score; verdict: string; decided_by: "fused" | "probe" };
export type ClaimResult = {
  attribute: string;
  claimed: string | number;
  p_match: number;
  verdict: "matches" | "mismatch";
  calibration: "calibrated" | "uncalibrated";
  clip_top: string;
  clip_p_claim: number;
  vlm_top: string | null;
  vlm_p_claim: number | null;
  options: string[];
  clip_dist: number[];
  vlm_dist: number[] | null;
};
export type VerifyResponse = {
  authenticity: Authenticity;
  consistency: ClaimResult[];
  claims: Record<string, string | number>;
  product_noun: string;
  jpeg_quality: number | null;
  image_size: number[];
  elapsed_s: number;
  vlm_used: boolean;
  mode: string;
  calibrator: string | null;
};
export type Example = {
  filename: string; description: string; product_noun: string; colour: string; material: string;
  count: string | number; truth_label: number; generator: string; mode: string; domain: string; image_key: string;
};
export type Health = { ok: boolean; models_loaded: boolean; vlm: boolean; calibrator: boolean; calibrator_file: string | null; mode: string; bundle: string };
export type Claims = { colour?: string; material?: string; count?: number };

const base = "";

export async function getHealth(): Promise<Health> {
  const r = await fetch(`${base}/api/health`);
  return r.json();
}
export async function getExamples(): Promise<Example[]> {
  const r = await fetch(`${base}/api/examples`);
  return r.ok ? r.json() : [];
}
export const exampleUrl = (filename: string) => `${base}/api/examples/${encodeURIComponent(filename)}`;

export async function extractClaims(description: string): Promise<Claims> {
  const fd = new FormData();
  fd.append("description", description);
  const r = await fetch(`${base}/api/claims`, { method: "POST", body: fd });
  return r.json();
}

export async function verify(file: Blob, description: string, noun: string, claims: Claims, jpeg: number | null) {
  const fd = new FormData();
  fd.append("image", file, "upload.jpg");
  fd.append("description", description);
  fd.append("product_noun", noun);
  fd.append("claims", JSON.stringify(claims));
  if (jpeg != null) fd.append("jpeg_quality", String(jpeg));
  const r = await fetch(`${base}/api/verify`, { method: "POST", body: fd });
  if (!r.ok) throw new Error(`verify failed: ${r.status} ${await r.text()}`);
  return (await r.json()) as VerifyResponse;
}

export const COLOURS = ["black", "white", "grey", "brown", "beige", "red", "orange", "yellow", "green", "blue", "purple", "pink"];
export const MATERIALS = ["wood", "metal", "plastic", "glass", "fabric", "leather", "ceramic", "stone"];
export const COUNTS = [1, 2, 3, 4, 5, 6];
