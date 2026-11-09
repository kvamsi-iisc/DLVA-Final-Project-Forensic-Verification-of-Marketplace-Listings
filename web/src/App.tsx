import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Activity, Loader2, Moon, ShieldCheck, Sun } from "lucide-react";
import { exampleUrl, extractClaims, getExamples, getHealth, verify, type Claims, type Example, type Health, type VerifyResponse } from "@/api";
import { AuthenticityCard } from "./components/AuthenticityCard";
import { ClaimsEditor } from "./components/ClaimsEditor";
import { ConsistencyCard } from "./components/ConsistencyCard";
import { Dropzone } from "./components/Dropzone";
import { ExamplesGallery } from "./components/ExamplesGallery";
import { JpegControl } from "./components/JpegControl";
import { Badge, Button, Card, CardContent, Separator } from "./components/ui";

function useTheme() {
  const [dark, setDark] = useState(() => {
    try { const s = localStorage.getItem("theme"); if (s) return s === "dark"; } catch { /* private mode */ }
    return window.matchMedia?.("(prefers-color-scheme: dark)").matches ?? false;
  });
  useEffect(() => {
    document.documentElement.classList.toggle("dark", dark);
    try { localStorage.setItem("theme", dark ? "dark" : "light"); } catch { /* ignore */ }
  }, [dark]);
  return { dark, toggle: () => setDark((d) => !d) };
}

export default function App() {
  const { dark, toggle } = useTheme();
  const [health, setHealth] = useState<Health | null>(null);
  const [examples, setExamples] = useState<Example[]>([]);
  const [selected, setSelected] = useState<Example | null>(null);
  const [file, setFile] = useState<Blob | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [description, setDescription] = useState("");
  const [noun, setNoun] = useState("");
  const [claims, setClaims] = useState<Claims>({});
  const [autoClaims, setAutoClaims] = useState<Claims>({});
  const [jpeg, setJpeg] = useState(95);
  const [result, setResult] = useState<VerifyResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [stale, setStale] = useState(false);          // inputs changed since the last verify
  const debounce = useRef<number>();
  const claimSeq = useRef(0);                          // drop out-of-order extraction responses
  const manualClaims = useRef(false);                  // a manual edit wins over auto-extraction

  useEffect(() => { getHealth().then(setHealth).catch(() => setHealth(null)); getExamples().then(setExamples).catch(() => {}); }, []);

  // Claim extraction mirrors the server's vocabulary; debounce so typing stays smooth.
  useEffect(() => {
    window.clearTimeout(debounce.current);
    const seq = ++claimSeq.current;
    manualClaims.current = false;
    debounce.current = window.setTimeout(() => {
      extractClaims(description).then((c) => {
        if (seq !== claimSeq.current || manualClaims.current) return;   // newer text or a manual edit arrived meanwhile
        setAutoClaims(c); setClaims(c);
      }).catch(() => {});
    }, 300);
  }, [description]);

  // Any input change after a result marks that result stale until Verify runs again.
  useEffect(() => { if (result) setStale(true); /* eslint-disable-next-line react-hooks/exhaustive-deps */ }, [description, noun, claims, jpeg, file]);

  const onFile = useCallback((f: File) => { setFile(f); setPreview(URL.createObjectURL(f)); setSelected(null); setResult(null); }, []);
  const clear = () => { setFile(null); setPreview(null); setSelected(null); setResult(null); };

  const pick = async (e: Example) => {
    const blob = await (await fetch(exampleUrl(e.filename))).blob();
    setFile(blob); setPreview(exampleUrl(e.filename)); setSelected(e); setResult(null);
    setDescription(e.description); setNoun(e.product_noun || "");
  };

  const run = async () => {
    if (!file) return;
    setLoading(true); setError(null);
    try { setResult(await verify(file, description, noun, claims, jpeg >= 95 ? null : jpeg)); setStale(false); }
    catch (err) { setError(String(err)); }
    finally { setLoading(false); }
  };

  const hasClaims = useMemo(() => Object.keys(claims).length > 0, [claims]);
  const truth = selected && result ? selected.truth_label : null;

  return (
    <div className="min-h-screen">
      <header className="sticky top-0 z-30 border-b bg-background/80 backdrop-blur supports-[backdrop-filter]:bg-background/60">
        <div className="container flex h-14 items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <div className="rounded-md bg-primary p-1.5 text-primary-foreground"><ShieldCheck className="h-4 w-4" /></div>
            <div className="leading-tight">
              <div className="text-sm font-semibold">Listing Verifier</div>
              <div className="hidden text-[11px] text-muted-foreground sm:block">One frozen CLIP ViT-L/14 · two small heads · CPU, offline</div>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <Badge variant={health?.models_loaded ? "success" : "warning"} className="hidden sm:inline-flex">
              <Activity className="h-3 w-3" />{health ? (health.models_loaded ? `${health.vlm ? "CLIP + SmolVLM" : "CLIP only"} · ${health.mode} mode${health.calibrator ? "" : " · uncalibrated claims"}` : "loading models") : "backend offline"}
            </Badge>
            <Button variant="ghost" size="icon" onClick={toggle} aria-label="toggle theme">{dark ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}</Button>
          </div>
        </div>
      </header>

      <main className="container space-y-6 py-6">
        <div className="grid gap-6 lg:grid-cols-5">
          <div className="space-y-6 lg:col-span-2">
            <Card>
              <CardContent className="space-y-5 p-5">
                <Dropzone file={file} previewUrl={preview} onFile={onFile} onClear={clear} />
                <ClaimsEditor description={description} onDescription={setDescription} noun={noun} onNoun={setNoun} claims={claims} onClaims={(c) => { manualClaims.current = true; setClaims(c); }} autoClaims={autoClaims} />
                <Separator />
                <JpegControl value={jpeg} onChange={setJpeg} />
                <Button size="lg" className="w-full" disabled={!file || loading} onClick={run}>
                  {loading ? <><Loader2 className="h-4 w-4 animate-spin" />Scoring on CPU…</> : "Verify listing"}
                </Button>
                {error && <p className="text-sm text-destructive">{error}</p>}
                {result && (
                  <p className="text-xs text-muted-foreground">
                    {result.elapsed_s}s · {result.image_size[0]}×{result.image_size[1]} px · {result.jpeg_quality ? `JPEG q=${result.jpeg_quality}` : "original"} · product “{result.product_noun}” · {result.mode} mode
                    {truth != null && <> · ground truth: <b className={truth ? "text-destructive" : "text-success"}>{truth ? "generated" : "real"}</b>{selected?.generator && truth ? ` (${selected.generator}, ${selected.mode})` : ""}</>}
                  </p>
                )}
              </CardContent>
            </Card>
          </div>
          <div className="space-y-6 lg:col-span-3">
            <AuthenticityCard a={result?.authenticity ?? null} loading={loading} stale={stale} />
            <ConsistencyCard rows={result?.consistency ?? null} loading={loading} hasClaims={hasClaims} />
          </div>
        </div>
        <ExamplesGallery examples={examples} selected={selected?.filename ?? null} onPick={pick} />
        <footer className="pb-6 pt-2 text-center text-[11px] text-muted-foreground">
          DLVA 2026 · IISc · Reals: ABO (CC BY 4.0) · Fakes: SDXL-Turbo (Stability AI Community License), SD 1.5 (CreativeML OpenRAIL-M) · Unseen test: CommunityForensics (CC BY-NC-SA 4.0)
        </footer>
      </main>
    </div>
  );
}
