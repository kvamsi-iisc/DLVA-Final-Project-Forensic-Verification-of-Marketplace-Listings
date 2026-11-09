import { Label } from "./ui";

export function JpegControl({ value, onChange }: { value: number; onChange: (v: number) => void }) {
  const none = value >= 95;
  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <Label htmlFor="jpeg">Simulate buyer-side compression</Label>
        <span className="font-mono text-xs tabular-nums text-muted-foreground">{none ? "original" : `JPEG q=${value}`}</span>
      </div>
      <input id="jpeg" type="range" min={30} max={95} step={5} value={value} onChange={(e) => onChange(Number(e.target.value))} />
      <p className="text-xs text-muted-foreground">
        Drag left to re-encode the photo the way a marketplace app or a screenshot would. Hypothesis under test: compression should hurt the frequency head more than the semantic probe. The measured curve is in the results, not in this sentence.
      </p>
    </div>
  );
}
