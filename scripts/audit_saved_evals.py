"""Audit saved eval outputs without making model calls or printing payloads."""

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from duplicate_side_effect_desk.reporting import (
    count_matching_rewards,
    format_distribution,
    reward_distribution,
    reward_is,
)

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
    if not ROOT.is_dir():
        sys.exit(
            "saved eval outputs are not present; this audit needs local outputs/evals. "
            "For public reproduction, run: python3 scripts/reproduce_model_table.py"
        )

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
        perfect = sum(reward_is(row["reward"], 1.0) for row in rows)
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

    print("\nsaved reward distributions by model and run")
    for model, run_dir, rows, metadata in sorted(all_runs, key=lambda item: (item[0], item[1].name)):
        split = split_name(len(rows))
        hints = metadata.get("env_args", {}).get("hints", False)
        non_error = [row for row in rows if row.get("error") is None]
        print(
            f"{model} {run_dir.name} {split} hints={hints} rows={len(rows)} "
            f"errors={len(rows) - len(non_error)} all=[{format_distribution(reward_distribution(rows))}] "
            f"non_error=[{format_distribution(reward_distribution(non_error)) or '-'}]"
        )

    print("\nsaved reward distributions by model and split")
    grouped = defaultdict(list)
    for model, _, rows, _ in all_runs:
        grouped[(model, split_name(len(rows)))].extend(rows)
    for (model, split), rows in sorted(grouped.items()):
        non_error = [row for row in rows if row.get("error") is None]
        print(
            f"{model} {split} rows={len(rows)} "
            f"all=[{format_distribution(reward_distribution(rows))}] "
            f"non_error=[{format_distribution(reward_distribution(non_error)) or '-'}]"
        )

    print("\nsaved reward distributions by split")
    for split in ("legacy-18", "current-32"):
        rows = [
            row
            for _, _, run_rows, _ in all_runs
            if split_name(len(run_rows)) == split
            for row in run_rows
        ]
        non_error = [row for row in rows if row.get("error") is None]
        print(
            f"{split} rows={len(rows)} all=[{format_distribution(reward_distribution(rows))}] "
            f"non_error=[{format_distribution(reward_distribution(non_error)) or '-'}]"
        )

    all_rows = [row for _, _, rows, _ in all_runs for row in rows]
    non_error_rows = [row for row in all_rows if row.get("error") is None]
    print("\nsaved reward distribution totals")
    print(f"all {len(all_rows)}: {format_distribution(reward_distribution(all_rows))}")
    print(
        f"excluding error rows {len(non_error_rows)}: "
        f"{format_distribution(reward_distribution(non_error_rows))}"
    )

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
    distribution = reward_distribution(selected)
    print("\nselected reward distribution")
    print(format_distribution(distribution))
    print(f"range: {min(distribution):.3f} to {max(distribution):.3f}")
    print(f"0.7 to 1.0: {sum(0.7 <= row['reward'] <= 1.0 for row in selected)}/{len(selected)}")
    print(
        "approximately 0.7 or exactly 1.0: "
        f"{count_matching_rewards(selected, (0.7, 1.0))}/{len(selected)}"
    )


def load_metadata(path):
    return json.loads(path.read_text(encoding="utf-8"))


def mean(rows):
    return sum(row["reward"] for row in rows) / len(rows)


def split_name(row_count):
    if row_count == 54:
        return "legacy-18"
    if row_count == 96:
        return "current-32"
    return f"other-{row_count}"


if __name__ == "__main__":
    main()
