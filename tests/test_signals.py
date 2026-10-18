import unittest

from pairs_trading.risk import max_holding_days_from_half_life
from pairs_trading.signals import FLAT, LONG_SPREAD, SHORT_SPREAD, PairStateMachine, SignalRules, decide_position

RULES = SignalRules(z_entry=2.0, z_exit=0.5, z_stop=3.5, cooldown_bars=3)


def decide(position, z, bars_held=1, max_hold=10, **kw):
    return decide_position(position, z, bars_held, max_hold, RULES, **kw)


class DecidePositionTests(unittest.TestCase):
    def test_enters_against_the_stretch(self) -> None:
        self.assertEqual(decide(FLAT, -2.1), (LONG_SPREAD, "entry"))
        self.assertEqual(decide(FLAT, 2.1), (SHORT_SPREAD, "entry"))

    def test_no_entry_inside_band_or_beyond_stop(self) -> None:
        self.assertEqual(decide(FLAT, 1.9), (FLAT, "flat"))
        self.assertEqual(decide(FLAT, 3.5), (FLAT, "flat"))
        self.assertEqual(decide(FLAT, -4.0), (FLAT, "flat"))
        self.assertEqual(decide(FLAT, float("nan")), (FLAT, "flat"))

    def test_entry_gates(self) -> None:
        self.assertEqual(decide(FLAT, -2.5, can_enter=False), (FLAT, "flat"))
        self.assertEqual(decide(FLAT, -2.5, force_reason="exit_square_off"), (FLAT, "flat"))

    def test_holds_while_still_stretched(self) -> None:
        self.assertEqual(decide(LONG_SPREAD, -1.5), (LONG_SPREAD, "hold"))
        self.assertEqual(decide(SHORT_SPREAD, 1.5), (SHORT_SPREAD, "hold"))
        self.assertEqual(decide(LONG_SPREAD, float("nan")), (LONG_SPREAD, "hold"))

    def test_takes_profit_on_reversion_including_overshoot(self) -> None:
        self.assertEqual(decide(LONG_SPREAD, -0.4), (FLAT, "exit_mean_reversion"))
        self.assertEqual(decide(LONG_SPREAD, 1.2), (FLAT, "exit_mean_reversion"))  # gapped through the mean
        self.assertEqual(decide(SHORT_SPREAD, 0.5), (FLAT, "exit_mean_reversion"))
        self.assertEqual(decide(SHORT_SPREAD, -2.0), (FLAT, "exit_mean_reversion"))

    def test_stops_only_on_adverse_extension(self) -> None:
        self.assertEqual(decide(LONG_SPREAD, -3.6), (FLAT, "exit_stop"))
        self.assertEqual(decide(SHORT_SPREAD, 3.5), (FLAT, "exit_stop"))

    def test_time_regime_and_forced_exits(self) -> None:
        self.assertEqual(decide(LONG_SPREAD, -1.5, bars_held=10, max_hold=10), (FLAT, "exit_time"))
        self.assertEqual(decide(LONG_SPREAD, -1.5, regime_ok=False), (FLAT, "exit_regime"))
        self.assertEqual(decide(LONG_SPREAD, -3.9, force_reason="exit_square_off"), (FLAT, "exit_square_off"))

    def test_rules_validate_threshold_order(self) -> None:
        with self.assertRaises(ValueError):
            SignalRules(z_entry=2.0, z_exit=2.5, z_stop=3.5)
        with self.assertRaises(ValueError):
            SignalRules(z_entry=2.0, z_exit=0.5, z_stop=1.5)


class StateMachineTests(unittest.TestCase):
    def machine(self) -> PairStateMachine:
        return PairStateMachine(RULES, max_hold_cap=20, default_hold=10)

    def test_cooldown_blocks_reentry_for_exactly_cooldown_bars_after_a_stop(self) -> None:
        m = self.machine()
        self.assertEqual(m.on_bar(-2.5, 5.0, 1.0), (LONG_SPREAD, "entry"))
        self.assertEqual(m.on_bar(-3.8, 5.0, 1.0), (FLAT, "exit_stop"))
        for _ in range(RULES.cooldown_bars):
            self.assertEqual(m.on_bar(-2.5, 5.0, 1.0), (FLAT, "flat"))
        self.assertEqual(m.on_bar(-2.5, 5.0, 1.0), (LONG_SPREAD, "entry"))

    def test_no_cooldown_after_a_profitable_exit(self) -> None:
        m = self.machine()
        m.on_bar(-2.5, 5.0, 1.0)
        self.assertEqual(m.on_bar(-0.2, 5.0, 1.0), (FLAT, "exit_mean_reversion"))
        self.assertEqual(m.on_bar(2.4, 5.0, 1.0), (SHORT_SPREAD, "entry"))

    def test_time_stop_uses_half_life_at_entry(self) -> None:
        m = self.machine()
        m.on_bar(-2.5, 2.0, 1.0)  # half-life 2 -> max hold 4 bars
        self.assertEqual(m.max_hold, 4)
        reasons = [m.on_bar(-1.5, 50.0, 1.0)[1] for _ in range(4)]  # later half-life must not extend it
        self.assertEqual(reasons, ["hold", "hold", "hold", "exit_time"])

    def test_refuses_entry_without_a_usable_hedge_ratio(self) -> None:
        m = self.machine()
        self.assertEqual(m.on_bar(-2.5, 5.0, -0.4), (FLAT, "flat"))
        self.assertEqual(m.on_bar(-2.5, 5.0, float("nan")), (FLAT, "flat"))


class RiskTests(unittest.TestCase):
    def test_max_hold_is_two_half_lives_capped(self) -> None:
        self.assertEqual(max_holding_days_from_half_life(3.2, cap=20), 7)
        self.assertEqual(max_holding_days_from_half_life(40.0, cap=20), 20)
        self.assertEqual(max_holding_days_from_half_life(float("nan"), cap=20), 10)
        self.assertEqual(max_holding_days_from_half_life(0.2, cap=20), 1)


if __name__ == "__main__":
    unittest.main()
