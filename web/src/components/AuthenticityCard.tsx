import { Brain, Layers, ShieldAlert, ShieldCheck, Waves } from "lucide-react";
import type { Authenticity, Score } from "@/api";
import { pct } from "@/lib/utils";
import { Badge, Card, CardContent, CardDescription, CardHeader, CardTitle, Meter, Skeleton } from "./ui";

function Row({ icon, name, hint, s }: { icon: React.ReactNode; name: string; hint: string; s: Score }) {
  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-2 text-sm font-medium">{icon}{name}</div>
        <div className="flex items-center gap-2">
          <span className="font-mono text-sm tabular-nums">{pct(s.p_fake)}</span>
          <Badge variant={s.flag ? "destructive" : "success"}>{s.flag ? "flagged" : "clear"}</Badge>
        </div>
      </div>
      <Meter value={s.p_fake} threshold={s.threshold} tone={s.flag ? "bad" : "good"} />
      <p className="text-xs text-muted-foreground">{hint}</p>
    </div>
  );
}

export function AuthenticityCard({ a, loading, stale }: { a: Authenticity | null; loading: boolean; stale?: boolean }) {
  const decider = a?.decided_by ?? "fused";
  const flagged = a ? a[decider]?.flag : false;
  return (
    <Card className="animate-fade-in">
      <CardHeader>
        <div className="flex items-start justify-between gap-3">
          <div className="space-y-1">
            <CardTitle>{flagged ? <ShieldAlert className="h-5 w-5 text-destructive" /> : <ShieldCheck className="h-5 w-5 text-success" />}Authenticity</CardTitle>
            <CardDescription>Real capture or diffusion-generated? Tick marks are the thresholds that flag 5% of real photos on validation.</CardDescription>
          </div>
          <div className="flex shrink-0 flex-col items-end gap-1">
            {a && <Badge variant={flagged ? "destructive" : "success"} className="text-sm">{a.verdict}</Badge>}
            {a && stale && <Badge variant="warning">inputs changed, re-verify</Badge>}
            {a && decider === "probe" && <Badge variant="outline">probe-only mode</Badge>}
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-5">
        {loading && !a ? (
          <><Skeleton className="h-10" /><Skeleton className="h-10" /><Skeleton className="h-10" /></>
        ) : a ? (
          <>
            {a.fused && <Row icon={<Layers className="h-4 w-4" />} name="Fused verdict" hint="Logistic fusion of the two branches, fitted on the calibration half of validation." s={a.fused} />}
            <Row icon={<Brain className="h-4 w-4" />} name="Semantic probe" hint="Linear probe on frozen CLIP ViT-L/14 features (UnivFD recipe)." s={a.probe} />
            {a.spectrum && <Row icon={<Waves className="h-4 w-4" />} name="Frequency head" hint="Small CNN on the log-DCT spectrum (AIDE-inspired cue). Score is a model output, not a calibrated probability." s={a.spectrum} />}
          </>
        ) : (
          <p className="text-sm text-muted-foreground">Upload a photo and press Verify.</p>
        )}
      </CardContent>
    </Card>
  );
}
