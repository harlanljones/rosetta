from __future__ import annotations

from rosetta.models.calibration import apply_affine_stat, calibration_applies, fit_prior_affine


def test_calibration_applies_only_aaa_path() -> None:
    assert calibration_applies(["NPB", "AAA", "MLB"], "MLB")
    assert not calibration_applies(["NPB", "MLB"], "MLB")


def test_fit_and_apply_affine() -> None:
    details = []
    for i in range(30):
        pred = 0.30 + i * 0.001
        details.append(
            {
                "role": "batter",
                "stat": "woba",
                "prediction": pred,
                "actual": pred * 0.9 + 0.02,
            }
        )
    fits = fit_prior_affine(details)
    assert "batter:woba" in fits
    center, low, high = apply_affine_stat("woba", 0.32, 0.28, 0.36, fits["batter:woba"])
    assert low < center < high
