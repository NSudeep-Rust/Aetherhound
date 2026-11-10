"""Tests for bearing estimation maths."""

import pytest
from capture.bearing import parabolic_offset


class TestParabolicOffset:
    """parabolic_offset is pure maths — no hardware, no config, trivially testable."""

    def test_peak_at_center(self):
        """Symmetric peak around y1 -> offset 0."""
        assert parabolic_offset(1.0, 2.0, 1.0) == 0.0

    def test_peak_left_of_center(self):
        """y0 > y2 with y1 as maximum -> negative offset."""
        assert parabolic_offset(4.0, 3.0, 1.0) == -0.5

    def test_peak_right_of_center(self):
        """y2 > y0 with y1 as maximum -> positive offset."""
        assert parabolic_offset(1.0, 3.0, 4.0) == 0.5

    def test_clipped_to_half_bin(self):
        """Extreme asymmetry clips to +/-0.5."""
        assert parabolic_offset(100.0, 1.0, 0.0) == 0.5
        assert parabolic_offset(0.0, 1.0, 100.0) == -0.5

    def test_flat_returns_zero(self):
        """Denominator near zero -> no curvature info -> 0."""
        assert parabolic_offset(1.0, 1.0, 1.0) == 0.0
        assert parabolic_offset(1.0, 1.0 + 1e-13, 1.0) == 0.0

    def test_known_geometry(self):
        """Parabola through (-1, y0), (0, y1), (1, y2) has vertex at x = -b/(2a)."""
        # y = (x - 0.25)^2 = x^2 - 0.5x + 0.0625, vertex at x=0.25
        y0 = 1.5625  # y(-1)
        y1 = 0.0625  # y(0)
        y2 = 0.5625  # y(1)
        assert abs(parabolic_offset(y0, y1, y2) - 0.25) < 1e-6

        # Vertex at -0.25: y = (x + 0.25)^2 = x^2 + 0.5x + 0.0625
        y0 = 0.5625  # y(-1)
        y1 = 0.0625  # y(0)
        y2 = 1.5625  # y(1)
        assert abs(parabolic_offset(y0, y1, y2) - (-0.25)) < 1e-6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])