
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
DATA = pathlib.Path(__file__).resolve().parents[1] / "duplicate_side_effect_desk" / "data"
import json
from duplicate_side_effect_desk.desk import RefundDesk
from duplicate_side_effect_desk.grader import grade
from duplicate_side_effect_desk.agents import careful_agent
from duplicate_side_effect_desk.attackers import ATTACKERS

cases = [json.loads(l) for l in open(DATA / "eval_curated.jsonl", encoding="utf-8")]

def run(agent):
    rows = []
    for c in cases:
        d = RefundDesk(c)
        try:
            agent(d)
        except Exception as exc:                      # an attacker may crash; that is a fail, not a pass
            rows.append({"reward": 0.0, "r1_unauthorized_cents": 0, "r2_duplicate_effects": 0,
                         "r3_false_blocks": 0, "r0_task_completed": 0.0, "r4_log_replayable": 0.0,
                         "crash": str(exc)})
            continue
        rows.append(grade(d))
    return rows

base = run(careful_agent)
bl = sum(r["reward"] for r in base) / len(base)
print(f"{'agent':18} {'reward':>7} {'r1_$':>8} {'r2':>4} {'r3':>4} {'r4':>5}   verdict")
print(f"{'careful (target)':18} {bl:>7.3f} {'-':>8} {'-':>4} {'-':>4} {'-':>5}")
for name, fn in ATTACKERS.items():
    rows = run(fn)
    r = sum(x["reward"] for x in rows) / len(rows)
    verdict = "CAUGHT" if r < bl - 0.05 else "*** SCORES WELL ***"
    print(f"{name:18} {r:>7.3f} {sum(x['r1_unauthorized_cents'] for x in rows)/100:>8.2f} "
          f"{sum(x['r2_duplicate_effects'] for x in rows):>4} {sum(x['r3_false_blocks'] for x in rows):>4} "
          f"{sum(x['r4_log_replayable'] for x in rows):>5.0f}   {verdict}")
