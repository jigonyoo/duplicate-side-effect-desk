from duplicate_side_effect_desk.reporting import (
    count_matching_rewards,
    reward_distribution,
    reward_is,
)


def test_decimal_tenths_are_compared_with_tolerance():
    rows = [
        {"reward": 0.7000000000000001},
        {"reward": 1.0},
        {"reward": 0.8},
    ]

    assert reward_is(0.7000000000000001, 0.7)
    assert count_matching_rewards(rows, (0.7, 1.0)) == 2
    assert reward_distribution(rows) == {0.7: 1, 0.8: 1, 1.0: 1}
