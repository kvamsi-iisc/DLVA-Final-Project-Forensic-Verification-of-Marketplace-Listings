import { exampleUrl, type Example } from "@/api";
import { cn } from "@/lib/utils";
import { Badge } from "./ui";

export function ExamplesGallery({ examples, selected, onPick }: { examples: Example[]; selected: string | null; onPick: (e: Example) => void }) {
  if (!examples.length) return null;
  return (
    <section className="space-y-2">
      <div className="flex items-baseline justify-between">
        <h2 className="text-sm font-semibold">Pre-tested inputs</h2>
        <p className="text-xs text-muted-foreground">Held-out listings; label shown after you verify.</p>
      </div>
      <div className="-mx-4 flex gap-3 overflow-x-auto px-4 pb-2 sm:mx-0 sm:grid sm:grid-cols-4 sm:overflow-visible sm:px-0 md:grid-cols-6 lg:grid-cols-8">
        {examples.map((e) => (
          <button
            key={e.filename}
            onClick={() => onPick(e)}
            className={cn("group relative w-28 shrink-0 overflow-hidden rounded-md border bg-muted/40 text-left transition-all hover:shadow-md sm:w-auto", selected === e.filename && "ring-2 ring-ring ring-offset-2 ring-offset-background")}
          >
            <img src={exampleUrl(e.filename)} alt={e.description} className="aspect-square w-full object-contain" loading="lazy" />
            <div className="truncate px-1.5 py-1 text-[11px] text-muted-foreground">{e.description}</div>
            {selected === e.filename && <Badge variant="secondary" className="absolute left-1 top-1">selected</Badge>}
          </button>
        ))}
      </div>
    </section>
  );
}
