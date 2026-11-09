import { useCallback, useRef, useState } from "react";
import { ImagePlus, X } from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "./ui";

export function Dropzone({ file, previewUrl, onFile, onClear }: { file: Blob | null; previewUrl: string | null; onFile: (f: File) => void; onClear: () => void }) {
  const [drag, setDrag] = useState(false);
  const input = useRef<HTMLInputElement>(null);
  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDrag(false);
      const f = e.dataTransfer.files?.[0];
      if (f && f.type.startsWith("image/")) onFile(f);
    },
    [onFile],
  );
  if (file && previewUrl) {
    return (
      <div className="relative overflow-hidden rounded-lg border bg-muted/40">
        <img src={previewUrl} alt="listing" className="mx-auto max-h-80 w-full object-contain" />
        <Button variant="secondary" size="icon" className="absolute right-2 top-2 h-8 w-8 rounded-full" onClick={onClear} aria-label="remove image">
          <X className="h-4 w-4" />
        </Button>
      </div>
    );
  }
  return (
    <div
      role="button"
      tabIndex={0}
      onClick={() => input.current?.click()}
      onKeyDown={(e) => e.key === "Enter" && input.current?.click()}
      onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
      onDragLeave={() => setDrag(false)}
      onDrop={onDrop}
      className={cn(
        "flex cursor-pointer flex-col items-center justify-center gap-2 rounded-lg border-2 border-dashed px-6 py-12 text-center transition-colors",
        drag ? "border-primary bg-accent" : "border-border hover:border-muted-foreground/50 hover:bg-accent/40",
      )}
    >
      <div className="rounded-full bg-secondary p-3"><ImagePlus className="h-6 w-6 text-muted-foreground" /></div>
      <p className="text-sm font-medium">Drop a listing photo here</p>
      <p className="text-xs text-muted-foreground">or click to choose a JPEG / PNG · stays on this machine</p>
      <input ref={input} type="file" accept="image/*" className="hidden" onChange={(e) => e.target.files?.[0] && onFile(e.target.files[0])} />
    </div>
  );
}
