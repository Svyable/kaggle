import pytest

from scripts.validate_e0_log import validate_log


def test_rejects_reset_only_false_green():
    text = """
No --game specified; playing all 2 games (this is what Kaggle does in competition rerun).

wa30 - RESET: count 0, levels completed 0, avg fps 0.0)
wa30 - RESET: count 1, levels completed 0, avg fps 10.0)
bp35 - RESET: count 0, levels completed 0, avg fps 0.0)
bp35 - RESET: count 1, levels completed 0, avg fps 10.0)
========= SUMMARY =========
  wa30     levels=  0  actions=   81  state=GameState.NOT_FINISHED
  bp35     levels=  0  actions=   81  state=GameState.NOT_FINISHED
"""
    with pytest.raises(ValueError, match="no non-RESET action"):
        validate_log(text)


def test_accepts_real_actions_even_without_level_progress():
    text = """
No --game specified; playing all 2 games (this is what Kaggle does in competition rerun).

wa30 - RESET: count 0, levels completed 0, avg fps 0.0)
wa30 - ACTION1: count 1, levels completed 0, avg fps 10.0)
bp35 - RESET: count 0, levels completed 0, avg fps 0.0)
bp35 - ACTION6: count 1, levels completed 0, avg fps 10.0)
========= SUMMARY =========
  wa30     levels=  0  actions=   81  state=GameState.NOT_FINISHED
  bp35     levels=  0  actions=   81  state=GameState.NOT_FINISHED
"""
    report = validate_log(text)
    assert report["games"] == 2
    assert report["levels_completed_total"] == 0


def test_smoke_can_require_exact_game_count():
    text = """
ls20 - RESET: count 0, levels completed 0, avg fps 0.0)
ls20 - ACTION4: count 1, levels completed 0, avg fps 10.0)
vc33 - RESET: count 0, levels completed 0, avg fps 0.0)
vc33 - ACTION1: count 1, levels completed 0, avg fps 10.0)
========= SUMMARY =========
  ls20     levels=  0  actions=   51  state=GameState.NOT_FINISHED
  vc33     levels=  0  actions=   51  state=GameState.NOT_FINISHED
"""
    assert validate_log(text, expected_games=2)["games"] == 2
    with pytest.raises(ValueError, match="expected 3"):
        validate_log(text, expected_games=3)
