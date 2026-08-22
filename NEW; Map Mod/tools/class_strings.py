from __future__ import annotations
import argparse, json, re, zipfile
from pathlib import Path

PRINTABLE = re.compile(rb"[\x20-\x7e]{4,}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jar")
    ap.add_argument("output")
    args = ap.parse_args()

    jar = Path(args.jar)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(jar, "r") as zf, out.open("w", encoding="utf-8") as f:
        for name in zf.namelist():
            if not (name.startswith("xaero/") and name.endswith(".class")):
                continue
            data = zf.read(name)
            strings = []
            for m in PRINTABLE.finditer(data):
                s = m.group().decode("ascii", errors="ignore")
                if s not in strings:
                    strings.append(s)
            obj = {
                "class": name[:-6].replace("/", "."),
                "strings": strings,
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
