"""Reproduce the committed model measurements without private eval outputs."""

import json
from collections import Counter, defaultdict
from pathlib import Path


REPORT = Path(__file__).resolve().parents[1] / "reports" / "model_eval_20260923.jsonl"
MODEL_ORDER = (
    "openai/gpt-4.1-mini",
    "anthropic/claude-haiku-4.5",
    "anthropic/claude-sonnet-4.5",
)
FAMILIES = ("timeout-then-retry", "legit-repeat-purchase")


def load_rows():
    return [
        json.loads(line)
        for line in REPORT.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def mean(rows):
    return sum(row["reward"] for row in rows) / len(rows)


def main():
    rows = load_rows()
    by_model = defaultdict(list)
    for row in rows:
        by_model[row["model"]].append(row)

    print("model                         mean    perfect   duplicates   unauthorized")
    for model in MODEL_ORDER:
        group = by_model[model]
        perfect = sum(row["reward"] == 1.0 for row in group)
        duplicate = int(sum(row["duplicate_effects"] for row in group))
        unauthorized = sum(row["unauthorized_cents"] for row in group) / 100
        print(
            f"{model:29} {mean(group):.3f}   {perfect:>2}/{len(group):<2}"
            f"      {duplicate:>2}          ${unauthorized:,.2f}"
        )

    print("\nfamily means")
    print("model                         timeout-then-retry   legit-repeat-purchase")
    for model in MODEL_ORDER:
        group = by_model[model]
        means = []
        for family in FAMILIES:
            family_rows = [row for row in group if row["family"] == family]
            means.append(mean(family_rows))
        print(f"{model:29} {means[0]:.3f}                {means[1]:.3f}")

    distribution = Counter(row["reward"] for row in rows)
    print("\nreward distribution")
    for reward in sorted(distribution):
        print(f"{reward:.3f}: {distribution[reward]}")
    in_band = sum(0.7 <= row["reward"] <= 1.0 for row in rows)
    exact = sum(row["reward"] in (0.7, 1.0) for row in rows)
    print(f"range: {min(distribution):.3f} to {max(distribution):.3f}")
    print(f"0.7 to 1.0: {in_band}/{len(rows)}")
    print(f"exactly 0.7 or 1.0: {exact}/{len(rows)}")


if __name__ == "__main__":
    main()
