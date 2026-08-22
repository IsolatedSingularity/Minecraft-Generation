from __future__ import annotations
import argparse, json, re
from pathlib import Path

# Heuristic parser. It is intentionally permissive because javap formatting
# differs slightly across JDK releases.
METHOD = re.compile(
    r"^\s*(?:public|protected|private|static|final|synchronized|native|abstract|strictfp|"
    r"default|\s)+\s*([^\s(]+)\s+([A-Za-z0-9_$<>]+)\(([^)]*)\)"
)

DESCRIPTOR = re.compile(r"^\s*descriptor:\s*(.+)$")
CALL = re.compile(r"//\s+(?:InterfaceMethod|Method)\s+([^:]+):(.+)$")

def parse_file(path: Path):
    text = path.read_text(encoding="utf-8", errors="replace").splitlines()
    cls = path.stem
    entries = []
    current = None

    for line in text:
        m = METHOD.match(line)
        if m:
            current = {
                "class_file": cls,
                "return_or_type": m.group(1),
                "method": m.group(2),
                "args": m.group(3),
                "descriptor": None,
                "calls": [],
            }
            entries.append(current)
            continue

        if current:
            d = DESCRIPTOR.match(line)
            if d and current["descriptor"] is None:
                current["descriptor"] = d.group(1).strip()

            c = CALL.search(line)
            if c:
                target = (c.group(1) + ":" + c.group(2)).strip()
                if target not in current["calls"]:
                    current["calls"].append(target)

    return entries

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("javap_dir")
    ap.add_argument("output")
    args = ap.parse_args()

    src = Path(args.javap_dir)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with out.open("w", encoding="utf-8") as f:
        for p in sorted(src.glob("*.txt")):
            for entry in parse_file(p):
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                count += 1

    print(f"Indexed {count} method-like entries -> {out}")

if __name__ == "__main__":
    main()
