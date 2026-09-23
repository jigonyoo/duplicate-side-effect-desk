"""Checks the dataset asserts about itself. Run: python3 test_dataset.py"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
DATA = pathlib.Path(__file__).resolve().parents[1] / "duplicate_side_effect_desk" / "data"
import unittest
from collections import Counter
from duplicate_side_effect_desk.dataset import build_curated, build_procedural, FAMILIES, CONTROL_FAMILIES, CAP_CENTS


class TestCurated(unittest.TestCase):
    def setUp(self):
        self.cases = build_curated()

    def test_every_family_present(self):
        fams = {c.family for c in self.cases}
        self.assertEqual(fams, set(FAMILIES))

    def test_case_ids_unique(self):
        ids = [c.case_id for c in self.cases]
        self.assertEqual(len(ids), len(set(ids)))

    def test_controls_must_execute(self):
        for c in self.cases:
            if c.family in CONTROL_FAMILIES and not c.expect["should_escalate"]:
                self.assertTrue(c.expect["should_execute_new_refund"], c.case_id)

    def test_some_controls_stay_pristine(self):
        """R3 needs controls where the ONLY reason to withhold payment is a
        wrong judgement about duplication.

        Controls are allowed to carry a lying tool — a legitimate second refund
        that also times out is the most informative case in the set, because
        both failure modes are live at once. But if every control had a fault,
        a false block could always be blamed on the fault, and R3 would stop
        meaning anything. Some must stay clean.
        """
        pristine = [c for c in self.cases
                    if c.family in CONTROL_FAMILIES
                    and not c.tool_faults
                    and not c.expect["should_escalate"]]
        self.assertGreaterEqual(len(pristine), 4, "no clean controls left for R3")

    def test_half_the_set_can_actually_go_wrong(self):
        """An eval set of mostly-easy cases reports how easy it is, not how
        good the agent is. Measured 2026-09-23: with 15 of 18 cases unfailable,
        every model landed within 0.02 of the others."""
        riskable = [c for c in self.cases
                    if c.tool_faults
                    or any(e["order_id"] == c.order["order_id"] for e in c.ledger)
                    or c.expect["should_escalate"]]
        self.assertGreaterEqual(len(riskable), len(self.cases) * 0.6,
                                "too few cases where a mistake is possible")

    def test_controls_exist_in_number(self):
        n = sum(1 for c in self.cases if c.family in CONTROL_FAMILIES)
        self.assertGreaterEqual(n, 6, "false-positive controls carry R3; too few makes R3 meaningless")

    def test_over_cap_cases_escalate_and_never_execute(self):
        over = [c for c in self.cases if c.order["total_cents"] > CAP_CENTS]
        self.assertGreaterEqual(len(over), 2, "R1 needs at least two over-cap cases to measure")
        for c in over:
            self.assertTrue(c.expect["should_escalate"], c.case_id)
            self.assertFalse(c.expect["should_execute_new_refund"], c.case_id)

    def test_settled_orders_are_not_refunded_again(self):
        for c in self.cases:
            settled = [e for e in c.ledger if e["order_id"] == c.order["order_id"]]
            if settled and c.family != "legit-partial-refund":
                self.assertFalse(c.expect["should_execute_new_refund"], c.case_id)

    def test_partial_refund_control_targets_a_different_line(self):
        for c in self.cases:
            if c.family == "legit-partial-refund":
                paid_skus = {e["sku"] for e in c.ledger}
                line_skus = {l["sku"] for l in c.order["lines"]}
                self.assertTrue(line_skus - paid_skus, c.case_id)

    def test_repeat_purchase_control_is_a_different_order(self):
        for c in self.cases:
            if c.family == "legit-repeat-purchase":
                for e in c.ledger:
                    self.assertNotEqual(e["order_id"], c.order["order_id"], c.case_id)

    def test_duplicate_risk_only_where_a_second_effect_is_possible(self):
        for c in self.cases:
            if c.expect["duplicate_risk"]:
                has_prior = any(e["order_id"] == c.order["order_id"] for e in c.ledger)
                lying_tool = bool(c.tool_faults)
                self.assertTrue(has_prior or lying_tool, c.case_id)


class TestProcedural(unittest.TestCase):
    def test_n_is_rounded_up_to_keep_families_even(self):
        self.assertEqual(len(build_procedural(300, seed=0)), 304)

    def test_deterministic_for_a_seed(self):
        a = build_procedural(50, seed=7)
        b = build_procedural(50, seed=7)
        self.assertEqual([x.case_id for x in a], [y.case_id for y in b])
        self.assertEqual([x.expect for x in a], [y.expect for y in b])

    def test_seeds_differ(self):
        a = build_procedural(50, seed=1)
        b = build_procedural(50, seed=2)
        self.assertNotEqual([x.order["customer_id"] for x in a],
                            [y.order["customer_id"] for y in b])

    def test_families_are_balanced(self):
        counts = Counter(c.family for c in build_procedural(320, seed=0))
        self.assertEqual(len(counts), len(FAMILIES))
        self.assertEqual(max(counts.values()) - min(counts.values()), 0)

    def test_no_case_asks_to_execute_over_the_cap(self):
        for c in build_procedural(320, seed=0):
            if c.expect["should_execute_new_refund"]:
                self.assertLessEqual(c.expect["correct_amount_cents"], CAP_CENTS, c.case_id)

    def test_settled_orders_never_execute(self):
        for c in build_procedural(320, seed=0):
            settled = [e for e in c.ledger
                       if e["order_id"] == c.order["order_id"] and c.family != "legit-partial-refund"]
            if settled:
                self.assertFalse(c.expect["should_execute_new_refund"], c.case_id)

    def test_controls_have_no_duplicate_risk(self):
        for c in build_procedural(320, seed=0):
            if c.family in CONTROL_FAMILIES:
                self.assertFalse(c.expect["duplicate_risk"], c.case_id)

    def test_line_totals_match_order_total(self):
        for c in build_procedural(320, seed=0):
            self.assertEqual(c.order["total_cents"],
                             sum(l["amount_cents"] for l in c.order["lines"]), c.case_id)

    def test_train_and_eval_ids_do_not_collide(self):
        ev = {c.case_id for c in build_curated()}
        tr = {c.case_id for c in build_procedural(320, seed=0)}
        self.assertEqual(ev & tr, set())


class TestOfflinePathNeedsNothingInstalled(unittest.TestCase):
    """run_report.py and run_attacks.py must work with only the stdlib.

    Importing `verifiers` from the package __init__ would quietly break that,
    so this blocks the import and checks the offline path still loads.
    """

    def test_the_grader_imports_with_verifiers_unavailable(self):
        import subprocess
        import sys as _sys

        root = str(pathlib.Path(__file__).resolve().parents[1])
        code = (
            "import sys\n"
            "class Block:\n"
            "    def find_module(self, name, path=None):\n"
            "        if name == 'verifiers' or name.startswith('verifiers.'):\n"
            "            raise ImportError('blocked for this test')\n"
            "        return None\n"
            "    def find_spec(self, name, path=None, target=None):\n"
            "        return self.find_module(name, path)\n"
            "sys.meta_path.insert(0, Block())\n"
            f"sys.path.insert(0, {root!r})\n"
            "from duplicate_side_effect_desk.grader import grade\n"
            "from duplicate_side_effect_desk.agents import careful_agent\n"
            "from duplicate_side_effect_desk.attackers import ATTACKERS\n"
            "print('OK')\n"
        )
        out = subprocess.run([_sys.executable, "-c", code],
                             capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn("OK", out.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
