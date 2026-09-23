"""Per-family breakdown of the most recent eval run for each model.

The headline reward averages over every family, so a model that fails badly on
the two families this environment is actually about can still post a 0.98. This
prints the number that matters.
"""
import json
import pathlib
import sys
from collections import defaultdict

ROOT = pathlib.Path(__file__).resolve().parents[1] / "outputs" / "evals"

def latest_per_model():
    out = {}
    for d in sorted(ROOT.iterdir()):
        if not d.is_dir():
            continue
        runs = [r for r in d.iterdir() if (r / "results.jsonl").exists()]
        if not runs:
            continue
        out[d.name] = max(runs, key=lambda r: r.stat().st_mtime)
    return out

def load(run):
    rows = []
    for line in (run / "results.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows

def main():
    runs = latest_per_model()
    if not runs:
        sys.exit("no results found")
    families = None
    table = {}
    for name, run in runs.items():
        rows = load(run)
        per = defaultdict(list)
        for r in rows:
            per[r["info"]["family"]].append(r)
        table[name] = per
        families = sorted(per) if families is None else families
    short = {n: n.split("--", 1)[1].replace("--", "/") for n in table}

    print(f"{'family':24}", end="")
    for n in table:
        print(f"{short[n][:20]:>22}", end="")
    print()
    print("-" * (24 + 22 * len(table)))
    for fam in families:
        print(f"{fam:24}", end="")
        for n in table:
            rows = table[n][fam]
            rew = sum(r["reward"] for r in rows) / len(rows)
            dup = sum(r["duplicate_effects"] for r in rows)
            print(f"{rew:>16.3f} {('dup' + str(int(dup))) if dup else '':>5}", end="")
        print()
    print("-" * (24 + 22 * len(table)))
    print(f"{'ALL':24}", end="")
    for n in table:
        rows = [r for fam in families for r in table[n][fam]]
        print(f"{sum(r['reward'] for r in rows)/len(rows):>16.3f} {'':>5}", end="")
    print()
    print()
    print("cases carrying the stale-read fault:")
    for n in table:
        rows = [r for fam in families for r in table[n][fam]]
        laggy = [r for r in rows if json.loads(r["info"]["case"])["tool_faults"].get("refund_history")]
        rew = sum(r["reward"] for r in laggy) / len(laggy)
        dup = sum(r["duplicate_effects"] for r in laggy)
        print(f"  {short[n]:32} n={len(laggy):3}  reward {rew:.3f}   duplicates {int(dup)}")

if __name__ == "__main__":
    main()
