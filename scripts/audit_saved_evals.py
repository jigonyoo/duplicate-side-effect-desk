"""Audit saved eval outputs without making model calls or printing payloads."""

import json
import re
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / "outputs" / "evals"
EXPECTED_ROWS = 96


def load(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def short_model(directory):
    return directory.split("duplicate-side-effect-desk--", 1)[1].replace("--", "/")


def main():
    all_runs = []
    for model_dir in sorted(path for path in ROOT.iterdir() if path.is_dir()):
        for run_dir in sorted(path for path in model_dir.iterdir() if path.is_dir()):
            results = run_dir / "results.jsonl"
            metadata = run_dir / "metadata.json"
            if results.exists() and metadata.exists():
                all_runs.append(
                    (short_model(model_dir.name), run_dir, load(results), load_metadata(metadata))
                )

    sizes = Counter(len(rows) for _, _, rows, _ in all_runs)
    print("run inventory")
    for size in sorted(sizes):
        print(f"{size}-row runs: {sizes[size]}")
    print(f"stored result rows: {sum(len(rows) for _, _, rows, _ in all_runs)}")

    official = {}
    for model, run_dir, rows, metadata in all_runs:
        if len(rows) != EXPECTED_ROWS:
            continue
        if metadata.get("env_args", {}).get("hints") is True:
            continue
        previous = official.get(model)
        if previous is None or run_dir.stat().st_mtime > previous[0].stat().st_mtime:
            official[model] = (run_dir, rows)

    print("\ncomparable 32-case runs (hints disabled)")
    for model in sorted(official):
        run_dir, rows = official[model]
        perfect = sum(row["reward"] == 1.0 for row in rows)
        duplicates = int(sum(row["duplicate_effects"] for row in rows))
        unauthorized = sum(row["unauthorized_cents"] for row in rows) / 100
        print(
            f"{model} {run_dir.name}: mean={mean(rows):.6f} "
            f"perfect={perfect}/{len(rows)} duplicates={duplicates} "
            f"unauthorized=${unauthorized:.2f}"
        )

    print("\nselected family means")
    for model in sorted(official):
        _, rows = official[model]
        by_family = defaultdict(list)
        for row in rows:
            by_family[row["info"]["family"]].append(row)
        for family in ("timeout-then-retry", "legit-repeat-purchase"):
            group = by_family[family]
            print(f"{model} {family}: n={len(group)} mean={mean(group):.6f}")

    error_rows = [
        row
        for _, _, rows, _ in all_runs
        for row in rows
        if row.get("error") is not None
    ]
    refused_402 = [row for row in error_rows if "402" in json.dumps(row["error"])]
    print("\nfailed stored rows")
    print(f"rows with error: {len(error_rows)}")
    print(f"rows whose error contains 402: {len(refused_402)}")
    print(f"402-row mean reward: {mean(refused_402):.12f}")
    print(f"402 rows with token_usage field: {sum('token_usage' in row for row in refused_402)}")

    orphan_count = 0
    orphan_files = []
    resultless_runs = sorted(
        path
        for model_dir in ROOT.iterdir()
        if model_dir.is_dir()
        for path in model_dir.iterdir()
        if path.is_dir() and not (path / "results.jsonl").exists()
    )
    for log in ROOT.rglob("env_worker_*.log"):
        if (log.parent / "results.jsonl").exists():
            continue
        count = sum(
            "402" in line
            for line in log.read_text(encoding="utf-8", errors="replace").splitlines()
        )
        if count:
            orphan_count += count
            orphan_files.append((log.relative_to(ROOT), count))
    print(f"run directories without results.jsonl: {len(resultless_runs)}")
    print(f"402 lines in logs without results.jsonl: {orphan_count}")
    print(f"refused-at-door lower bound: {len(refused_402) + orphan_count}")
    for path, count in orphan_files:
        print(f"orphan log: {path} ({count})")

    patterns = {
        "email": re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I),
        "url": re.compile(r"https?://", re.I),
        "ipv4": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
        "openai_key": re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
        "aws_key": re.compile(r"\bAKIA[A-Z0-9]{16}\b"),
        "github_token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
        "bearer": re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{16,}\b", re.I),
    }
    scan_counts = Counter()
    for _, _, rows, _ in all_runs:
        if len(rows) != EXPECTED_ROWS:
            continue
        for row in rows:
            serialized = json.dumps(row, ensure_ascii=False)
            for name, pattern in patterns.items():
                scan_counts[name] += len(pattern.findall(serialized))
    print("\nprivacy scan over all four 96-row files (counts only)")
    print(" ".join(f"{name}={scan_counts[name]}" for name in patterns))

    selected = [row for _, rows in official.values() for row in rows]
    distribution = Counter(row["reward"] for row in selected)
    print("\nselected reward distribution")
    print(" ".join(f"{score:.3f}:{distribution[score]}" for score in sorted(distribution)))
    print(f"range: {min(distribution):.3f} to {max(distribution):.3f}")
    print(f"0.7 to 1.0: {sum(0.7 <= row['reward'] <= 1.0 for row in selected)}/{len(selected)}")
    print(f"exactly 0.7 or 1.0: {sum(row['reward'] in (0.7, 1.0) for row in selected)}/{len(selected)}")


def load_metadata(path):
    return json.loads(path.read_text(encoding="utf-8"))


def mean(rows):
    return sum(row["reward"] for row in rows) / len(rows)


if __name__ == "__main__":
    main()
