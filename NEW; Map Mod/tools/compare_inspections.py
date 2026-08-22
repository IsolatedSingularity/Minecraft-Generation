from __future__ import annotations
import argparse, json
from pathlib import Path

KEYWORDS = (
    "MapWriter", "MapPixel", "MapProcessor", "MapRegion", "MapTile",
    "Overlay", "Cave", "Biome", "Color", "Colour", "Light",
    "Height", "Texture", "Cache", "WorldData", "RegionDetection", "GuiMap"
)

def load_classes(root: Path):
    p = root / "xaero_classes.txt"
    if not p.exists():
        return set()
    return {x.strip() for x in p.read_text(encoding="utf-8", errors="replace").splitlines() if x.strip()}

def load_methods(root: Path):
    p = root / "class_methods.jsonl"
    out = []
    if not p.exists():
        return out
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            out.append(json.loads(line))
        except Exception:
            pass
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("output")
    args = ap.parse_args()

    a = Path(args.a)
    b = Path(args.b)
    out = Path(args.output)

    ca, cb = load_classes(a), load_classes(b)
    only_a = sorted(ca - cb)
    only_b = sorted(cb - ca)
    shared = sorted(ca & cb)

    interesting_a = sorted(c for c in ca if any(k.lower() in c.lower() for k in KEYWORDS))
    interesting_b = sorted(c for c in cb if any(k.lower() in c.lower() for k in KEYWORDS))

    lines = [
        "# Xaero Inspection Comparison",
        "",
        f"- A: `{a}`",
        f"- B: `{b}`",
        f"- classes A: {len(ca)}",
        f"- classes B: {len(cb)}",
        f"- shared: {len(shared)}",
        "",
        "## Interesting classes in A",
        ""
    ]
    lines += [f"- `{x}`" for x in interesting_a]
    lines += ["", "## Interesting classes in B", ""]
    lines += [f"- `{x}`" for x in interesting_b]
    lines += ["", "## Only in A", ""]
    lines += [f"- `{x}`" for x in only_a]
    lines += ["", "## Only in B", ""]
    lines += [f"- `{x}`" for x in only_b]

    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out}")

if __name__ == "__main__":
    main()
