"""
Pythagorean Win Expectation baseline.

Bill James's formula:  W_est = 162 * R^2 / (R^2 + RA^2)

Uses only runs scored (R) and runs allowed (RA) — no learning involved.
This serves as the naive baseline that the ML models aim to beat.
"""
