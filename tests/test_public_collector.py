from types import SimpleNamespace

import pytest

from research.public_collector import transition_from_step


def frame(grid, *, levels=0, state="NOT_FINISHED"):
    return SimpleNamespace(
        frame=[grid],
        levels_completed=levels,
        state=SimpleNamespace(value=state),
        game_id="demo-version",
    )


def action(action_id, *, x=None, y=None):
    data = SimpleNamespace()
    if x is not None:
        data.x = x
    if y is not None:
        data.y = y
    return SimpleNamespace(value=action_id, action_data=data)


def test_transition_is_action_aligned_and_counts_change():
    t = transition_from_step(
        game_family="demo",
        previous_grid=[[0, 0]],
        current_frame=frame([[0, 1]], levels=0),
        action=action(1),
        next_frame=frame([[0, 2]], levels=1),
    )
    assert t is not None
    assert t.action_id == 1
    assert t.previous_grid == [[0, 0]]
    assert t.current_grid == [[0, 1]]
    assert t.next_grid == [[0, 2]]
    assert t.changed_cells == 1
    assert t.progressed is True
    assert t.died is False


def test_reset_is_skipped():
    assert (
        transition_from_step(
            game_family="demo",
            previous_grid=[],
            current_frame=frame([]),
            action=action(0),
            next_frame=frame([[0]]),
        )
        is None
    )


def test_action6_coordinates_are_preserved():
    t = transition_from_step(
        game_family="demo",
        previous_grid=[[0]],
        current_frame=frame([[0]]),
        action=action(6, x=17, y=23),
        next_frame=frame([[1]]),
    )
    assert t is not None
    assert (t.click_x, t.click_y) == (17, 23)


def test_action6_bad_coordinate_is_rejected():
    with pytest.raises(ValueError, match="out of bounds"):
        transition_from_step(
            game_family="demo",
            previous_grid=[[0]],
            current_frame=frame([[0]]),
            action=action(6, x=64, y=0),
            next_frame=frame([[1]]),
        )
