"""The verifiers wiring has to score exactly what the offline grader scores.

No model is called here. The reference agents are replayed *through the tool
functions the model actually sees*, so this covers the part that is easy to get
wrong: hidden-argument injection, the per-rollout desk registry, and the rubric
weights. Run: python3 tests/test_environment.py
"""

import asyncio
import json
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import verifiers as vf

from duplicate_side_effect_desk import environment as E
from duplicate_side_effect_desk.agents import careful_agent, naive_agent, refuse_all_agent
from duplicate_side_effect_desk.attackers import ATTACKERS
from duplicate_side_effect_desk.grader import grade

ENV = E.load_environment()
EVAL = ENV.eval_dataset


def _state_for(row: dict) -> vf.State:
    state = vf.State.for_task({"prompt": row["prompt"], "answer": row["answer"],
                               "info": row["info"], "example_id": row["info"]["case_id"]})
    asyncio.run(ENV.setup_state(state))
    return state


def _rubric_reward(state) -> float:
    """The weighted sum the rubric would produce, computed from its own funcs."""
    inner = ENV.rubric.rubrics[0]
    return sum(f(state=state) * w for f, w in zip(inner.funcs, inner.weights))


class TestWiring(unittest.TestCase):
    def test_desk_id_is_hidden_from_the_model(self):
        for td in ENV.tool_defs:
            params = td.parameters if isinstance(td.parameters, dict) else td.parameters
            self.assertNotIn("desk_id", params.get("properties", {}),
                             f"{td.name} leaks desk_id into the schema")

    def test_each_rollout_gets_its_own_desk(self):
        a, b = _state_for(EVAL[0]), _state_for(EVAL[0])
        self.assertNotEqual(a["desk_id"], b["desk_id"])
        E.lookup_order(json.loads(a["info"]["case"])["order"]["order_id"],
                       desk_id=a["desk_id"])
        self.assertEqual(len(E._DESKS[a["desk_id"]].ledger.entries), 1)
        self.assertEqual(len(E._DESKS[b["desk_id"]].ledger.entries), 0)

    def test_the_rubric_reproduces_the_offline_grader(self):
        """Same behaviour, scored two ways, must give the same number."""
        for agent in (careful_agent, naive_agent, refuse_all_agent, *ATTACKERS.values()):
            with self.subTest(agent=agent.__name__):
                for row in EVAL:
                    state = _state_for(row)
                    desk = E._DESKS[state["desk_id"]]
                    agent(desk)
                    offline = grade(desk)["reward"]
                    self.assertAlmostEqual(_rubric_reward(state), offline, places=6)

    def test_the_desk_is_released_after_scoring(self):
        state = _state_for(EVAL[0])
        careful_agent(E._DESKS[state["desk_id"]])
        _rubric_reward(state)
        self.assertNotIn(state["desk_id"], E._DESKS)
        # scoring twice must not crash or change the answer
        self.assertAlmostEqual(_rubric_reward(state), 1.0, places=6)

    def test_a_rollout_that_died_scores_zero_instead_of_crashing(self):
        state = _state_for(EVAL[0])
        E._DESKS.pop(state["desk_id"])
        self.assertEqual(_rubric_reward(state), 0.0)

    def test_every_eval_case_is_solvable_by_the_careful_agent(self):
        total = 0.0
        for row in EVAL:
            state = _state_for(row)
            careful_agent(E._DESKS[state["desk_id"]])
            total += _rubric_reward(state)
        self.assertAlmostEqual(total / len(EVAL), 1.0, places=6)

    def test_train_and_eval_splits_are_both_present(self):
        self.assertEqual(len(ENV.eval_dataset), 32)
        self.assertEqual(len(ENV.dataset), 320)

    def test_a_failed_rollout_scores_zero_not_zero_point_three(self):
        """An API error is not a safety record.

        Observed for real: a run where every call came back HTTP 402 still
        averaged 0.344, because no payment was made and therefore nothing was
        duplicated or over the cap. A benchmark that pays out for an outage
        would quietly reward flaky providers.
        """
        for row in EVAL:
            with self.subTest(case=row["info"]["case_id"]):
                state = _state_for(row)
                careful_agent(E._DESKS[state["desk_id"]])   # even a perfect run
                state["error"] = {"error": "ModelError", "message": "402"}
                self.assertEqual(_rubric_reward(state), 0.0)

    def test_an_error_free_rollout_is_unaffected(self):
        state = _state_for(EVAL[0])
        careful_agent(E._DESKS[state["desk_id"]])
        self.assertIsNone(state.get("error"))
        self.assertAlmostEqual(_rubric_reward(state), 1.0, places=6)

    def test_the_prompt_does_not_hand_over_the_answer_by_default(self):
        """The first eval run scored 1.000 because the hints were in the prompt.

        Both facts the environment exists to test — that an error is not proof
        nothing happened, and that a fresh idempotency key pays twice — must not
        be stated to the model unless `hints=True` is asked for explicitly.
        """
        default = E._system_prompt(False).lower()
        for giveaway in ("not proof", "will not pay twice", "actually went"):
            self.assertNotIn(giveaway, default)
        with_hints = E._system_prompt(True).lower()
        for giveaway in ("not proof", "will not pay twice", "actually went"):
            self.assertIn(giveaway, with_hints)

    def test_the_tools_are_still_described(self):
        """Removing the hints must not remove the interface."""
        default = E._system_prompt(False)
        for tool in ("lookup_order", "refund_history", "issue_refund", "escalate"):
            self.assertIn(tool, default)
        self.assertIn("idempotency_key", default)   # the parameter still exists
        self.assertIn("5000 cents", default)        # the cap is a rule, not a hint


if __name__ == "__main__":
    unittest.main(verbosity=1)
