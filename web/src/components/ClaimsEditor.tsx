import { COLOURS, COUNTS, MATERIALS, type Claims } from "@/api";
import { Badge, Input, Label, Select, Textarea } from "./ui";
import { Sparkles } from "lucide-react";

type Props = {
  description: string; onDescription: (s: string) => void;
  noun: string; onNoun: (s: string) => void;
  claims: Claims; onClaims: (c: Claims) => void;
  autoClaims: Claims;
};

export function ClaimsEditor({ description, onDescription, noun, onNoun, claims, onClaims, autoClaims }: Props) {
  const set = (k: keyof Claims, v: string) => {
    const next: Claims = { ...claims };
    if (v === "") delete next[k];
    else if (k === "count") next.count = Number(v);
    else next[k] = v;
    onClaims(next);
  };
  const n = Object.keys(autoClaims).length;
  return (
    <div className="space-y-4">
      <div className="space-y-1.5">
        <Label htmlFor="desc">Seller description</Label>
        <Textarea id="desc" value={description} onChange={(e) => onDescription(e.target.value)} placeholder="e.g. Brown leather sofa, set of 2" />
        <p className="flex items-center gap-1.5 text-xs text-muted-foreground">
          <Sparkles className="h-3.5 w-3.5" />
          {n ? <>Found {n} checkable claim{n > 1 ? "s" : ""}: {Object.entries(autoClaims).map(([k, v]) => <Badge key={k} variant="secondary">{k}: {String(v)}</Badge>)}</> : "Only colour, material and item count are checkable; other words are ignored. Negations are handled only in the form 'not X'. Edit the selectors below to set the claim exactly."}
        </p>
      </div>
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <div className="col-span-2 space-y-1.5 sm:col-span-1">
          <Label htmlFor="noun">Product</Label>
          <Input id="noun" value={noun} onChange={(e) => onNoun(e.target.value)} placeholder="sofa" className="h-9" />
        </div>
        <div className="space-y-1.5">
          <Label>Colour</Label>
          <Select value={claims.colour ?? ""} onChange={(e) => set("colour", e.target.value)} className="w-full">
            <option value="">— none —</option>
            {COLOURS.map((c) => <option key={c} value={c}>{c}</option>)}
          </Select>
        </div>
        <div className="space-y-1.5">
          <Label>Material</Label>
          <Select value={claims.material ?? ""} onChange={(e) => set("material", e.target.value)} className="w-full">
            <option value="">— none —</option>
            {MATERIALS.map((c) => <option key={c} value={c}>{c}</option>)}
          </Select>
        </div>
        <div className="space-y-1.5">
          <Label>Count</Label>
          <Select value={claims.count ?? ""} onChange={(e) => set("count", e.target.value)} className="w-full">
            <option value="">— none —</option>
            {COUNTS.map((c) => <option key={c} value={c}>{c}</option>)}
          </Select>
        </div>
      </div>
    </div>
  );
}
