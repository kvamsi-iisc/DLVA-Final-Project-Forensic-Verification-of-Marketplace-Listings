import { useState } from "react";
import { CheckCircle2, ChevronDown, ListChecks, XCircle } from "lucide-react";
import type { ClaimResult } from "@/api";
import { cn, pct, titleCase } from "@/lib/utils";
import { Badge, Card, CardContent, CardDescription, CardHeader, CardTitle, Meter, Skeleton } from "./ui";

function Dist({ label, options, dist, claimIdx }: { label: string; options: string[]; dist: number[]; claimIdx: number }) {
  const max = Math.max(...dist, 1e-6);
  return (
    <div className="space-y-1">
      <p className="text-xs font-medium text-muted-foreground">{label}</p>
      <div className="grid grid-cols-2 gap-x-4 gap-y-1 sm:grid-cols-3">
        {options.map((o, i) => (
          <div key={o} className="flex items-center gap-2 text-xs">
            <span className={cn("w-16 truncate", i === claimIdx && "font-semibold text-foreground")}>{o}</span>
            <div className="h-1.5 flex-1 rounded-full bg-secondary"><div className={cn("h-full rounded-full", i === claimIdx ? "bg-primary" : "bg-muted-foreground/50")} style={{ width: `${(dist[i] / max) * 100}%` }} /></div>
            <span className="w-9 text-right font-mono tabular-nums text-muted-foreground">{pct(dist[i])}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function ClaimRow({ c }: { c: ClaimResult }) {
  const [open, setOpen] = useState(false);
  const ok = c.verdict === "matches";
  const idx = c.options.indexOf(String(c.claimed));
  return (
    <div className="rounded-md border p-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          {ok ? <CheckCircle2 className="h-4 w-4 text-success" /> : <XCircle className="h-4 w-4 text-destructive" />}
          <span className="text-sm"><span className="text-muted-foreground">{titleCase(c.attribute)}:</span> <span className="font-semibold">{String(c.claimed)}</span></span>
          {c.calibration === "uncalibrated" && <Badge variant="outline">uncalibrated</Badge>}
        </div>
        <div className="flex items-center gap-2">
          <span className="font-mono text-sm tabular-nums">{pct(c.p_match)}</span>
          <Badge variant={ok ? "success" : "destructive"}>{ok ? "matches" : "mismatch"}</Badge>
        </div>
      </div>
      <Meter value={c.p_match} threshold={0.5} tone={ok ? "good" : "bad"} className="mt-2" />
      <div className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-muted-foreground">
        <span>CLIP sees <b className="text-foreground">{c.clip_top}</b> ({pct(c.clip_p_claim)} on claim)</span>
        {c.vlm_top != null && <span>SmolVLM picks <b className="text-foreground">{c.vlm_top}</b> ({pct(c.vlm_p_claim)} on claim)</span>}
        <button onClick={() => setOpen(!open)} className="ml-auto inline-flex items-center gap-1 hover:text-foreground">
          details <ChevronDown className={cn("h-3.5 w-3.5 transition-transform", open && "rotate-180")} />
        </button>
      </div>
      {open && (
        <div className="mt-3 space-y-3 border-t pt-3 animate-fade-in">
          <Dist label="CLIP zero-shot over options" options={c.options} dist={c.clip_dist} claimIdx={idx} />
          {c.vlm_dist && <Dist label="SmolVLM multiple-choice over options" options={c.options} dist={c.vlm_dist} claimIdx={idx} />}
        </div>
      )}
    </div>
  );
}

export function ConsistencyCard({ rows, loading, hasClaims }: { rows: ClaimResult[] | null; loading: boolean; hasClaims: boolean }) {
  const bad = rows?.filter((r) => r.verdict === "mismatch").length ?? 0;
  return (
    <Card className="animate-fade-in">
      <CardHeader>
        <div className="flex items-start justify-between gap-3">
          <div className="space-y-1">
            <CardTitle><ListChecks className="h-5 w-5" />Description consistency</CardTitle>
            <CardDescription>Each checkable claim is scored by CLIP zero-shot contrast and a multiple-choice question to SmolVLM, then calibrated on synthetic mismatches.</CardDescription>
          </div>
          {rows && rows.length > 0 && <Badge variant={bad ? "warning" : "success"} className="shrink-0 text-sm">{bad ? `${bad} mismatch${bad > 1 ? "es" : ""}` : "all claims match"}</Badge>}
        </div>
      </CardHeader>
      <CardContent className="space-y-3">
        {loading && !rows ? (
          <><Skeleton className="h-16" /><Skeleton className="h-16" /></>
        ) : rows && rows.length ? (
          rows.map((c) => <ClaimRow key={c.attribute} c={c} />)
        ) : (
          <p className="text-sm text-muted-foreground">{hasClaims ? "Press Verify to score the claims." : "No checkable claim in the description. Add a colour, material or item count."}</p>
        )}
      </CardContent>
    </Card>
  );
}
