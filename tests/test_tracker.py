"""Unit tests for TrajectoryStateTracker and the delusion_slope metric."""

import pytest

from psychosis_guard.tracker import TrajectoryState, TrajectoryStateTracker


class TestDelusionSlope:
    def test_fewer_than_two_points_is_zero(self):
        state = TrajectoryState()
        assert state.delusion_slope() == 0.0
        state.delusion_density.append(0.7)
        assert state.delusion_slope() == 0.0

    def test_flat_series_is_zero(self):
        state = TrajectoryState(delusion_density=[0.4, 0.4, 0.4, 0.4])
        assert state.delusion_slope() == pytest.approx(0.0)

    def test_known_linear_series(self):
        # y = 0.25 * x  ->  slope exactly 0.25
        state = TrajectoryState(delusion_density=[0.0, 0.25, 0.5, 0.75, 1.0])
        assert state.delusion_slope() == pytest.approx(0.25)

    def test_descending_series_is_negative(self):
        state = TrajectoryState(delusion_density=[0.9, 0.6, 0.3])
        assert state.delusion_slope() < 0

    def test_windowed_slope_sees_recent_trend(self):
        # long suppressed stretch, then a sharp recent climb: the whole-series
        # slope stays numb while the windowed slope reacts — this reactivity
        # is why the policy reads the windowed variant
        state = TrajectoryState(delusion_density=[0.2] * 10 + [0.3, 0.5, 0.7, 0.9])
        assert state.delusion_slope(window=4) == pytest.approx(0.2)
        assert state.delusion_slope() < state.delusion_slope(window=4)

    def test_window_larger_than_series_equals_full(self):
        state = TrajectoryState(delusion_density=[0.1, 0.5, 0.6])
        assert state.delusion_slope(window=10) == state.delusion_slope()


class TestTracker:
    def test_update_accumulates_observations(self):
        tracker = TrajectoryStateTracker()
        tracker.update("t1", delusion_density=0.2, conviction=0.3,
                       bot_validated=True, isolation=False)
        tracker.update("t2", delusion_density=0.6, conviction=0.5,
                       bot_validated=True, isolation=True)
        tracker.update("t3", delusion_density=0.8, conviction=0.9,
                       bot_validated=False, isolation=False)

        s = tracker.state
        assert s.turn_count == 3
        assert s.delusion_density == [0.2, 0.6, 0.8]
        assert s.conviction_series == [0.3, 0.5, 0.9]
        assert s.bot_validation_count == 2
        assert s.isolation_mentions == 1
        assert s.delusion_slope() == pytest.approx(0.3)
