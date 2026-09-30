#!/usr/bin/env python3
"""Extract the authoritative corpus numbering: file -> declared doc number + title line."""
import re, glob, os

DL = "/home/z/my-project/download"
ORD = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6,
    "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10, "eleventh": 11,
    "twelfth": 12, "thirteenth": 13, "fourteenth": 14, "fifteenth": 15,
    "sixteenth": 16, "seventeenth": 17, "eighteenth": 18, "nineteenth": 19,
    "twentieth": 20,
}
TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
        "seventy": 70, "eighty": 80, "ninety": 90}

def ordinal_to_int(s):
    s = s.strip().lower().replace("-", " ")
    parts = s.split()
    if len(parts) == 1:
        if parts[0] in ORD:
            return ORD[parts[0]]
        return None
    # e.g. "eighty second"
    total = 0
    for p in parts:
        if p in TENS:
            total += TENS[p]
        elif p in ORD:
            total += ORD[p]
        else:
            return None
    return total if total else None

rows = []
for path in sorted(glob.glob(os.path.join(DL, "*.md"))):
    fn = os.path.basename(path)
    if fn == "README.md":
        continue
    with open(path, encoding="utf-8") as f:
        head = f.read(12000)
    head = head.replace("\u2019", "'").replace("\u2018", "'")
    m = re.search(r"corpus's ([a-z\- ]+?) document", head)
    num = ordinal_to_int(m.group(1)) if m else None
    t = re.search(r"^#\s+(.+)$", head, re.M)
    title = t.group(1).strip() if t else "(no title)"
    rows.append((num if num is not None else -1, fn, title))

rows.sort(key=lambda r: r[0])
out = []
for num, fn, title in rows:
    tag = str(num) if num > 0 else "??"
    out.append(f"{tag:>3}  {fn:<50}  {title[:110]}")
report = "\n".join(out)
with open("/home/z/my-project/tool-results/doc_index.txt", "w") as f:
    f.write(report + "\n")
print(report)
print(f"\nTotal: {len(rows)} docs; unresolved: {sum(1 for r in rows if r[0] < 0)}")
