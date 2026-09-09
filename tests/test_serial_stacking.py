import math


def apply_efficiency(current_load, treated_fraction, efficiency):
    """Reference implementation of the documented serial-stacking rule.

    This mirrors the documented behavior:

        L_new = L_old × (1 - f_t × e)

    where:
    - L_old is the current remaining load
    - f_t is the treated fraction
    - e is the sampled BMP efficiency

    It is used here as a behavioral contract test for the model's stated
    assumptions, especially that later BMPs act on the current remaining load.
    """
    return current_load * (1.0 - treated_fraction * efficiency)


def test_two_full_50_percent_reductions_produce_75_percent_total_reduction():
    load0 = 100.0

    load1 = apply_efficiency(load0, treated_fraction=1.0, efficiency=0.50)
    load2 = apply_efficiency(load1, treated_fraction=1.0, efficiency=0.50)

    assert math.isclose(load1, 50.0, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(load2, 25.0, rel_tol=0.0, abs_tol=1e-12)


def test_second_bmp_acts_on_remaining_load_not_original_load():
    load0 = 100.0

    load_after_first = apply_efficiency(load0, treated_fraction=1.0, efficiency=0.40)
    load_after_second = apply_efficiency(load_after_first, treated_fraction=1.0, efficiency=0.25)

    expected = 100.0 * (1.0 - 0.40) * (1.0 - 0.25)

    assert math.isclose(load_after_first, 60.0, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(load_after_second, expected, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(load_after_second, 45.0, rel_tol=0.0, abs_tol=1e-12)


def test_partial_treatment_fraction_reduces_only_treated_portion():
    load0 = 100.0

    load1 = apply_efficiency(load0, treated_fraction=0.50, efficiency=0.40)

    # 50% of the load is treated at 40% efficiency => total reduction = 20%
    assert math.isclose(load1, 80.0, rel_tol=0.0, abs_tol=1e-12)


def test_negative_efficiency_increases_load():
    load0 = 100.0

    load1 = apply_efficiency(load0, treated_fraction=1.0, efficiency=-0.20)

    assert math.isclose(load1, 120.0, rel_tol=0.0, abs_tol=1e-12)


def test_failure_adjusted_efficiency_scales_effect_before_application():
    load0 = 100.0
    sampled_efficiency = 0.50
    failure_reduction = 0.25

    failed_efficiency = sampled_efficiency * failure_reduction
    load1 = apply_efficiency(load0, treated_fraction=1.0, efficiency=failed_efficiency)

    # 50% sampled efficiency with failure reduction 0.25 => effective efficiency 12.5%
    assert math.isclose(failed_efficiency, 0.125, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(load1, 87.5, rel_tol=0.0, abs_tol=1e-12)


def test_serial_stacking_with_failure_then_normal_application():
    load0 = 100.0

    first_eff = 0.50 * 0.25   # failed BMP
    second_eff = 0.50         # normal BMP

    load1 = apply_efficiency(load0, treated_fraction=1.0, efficiency=first_eff)
    load2 = apply_efficiency(load1, treated_fraction=1.0, efficiency=second_eff)

    expected = 100.0 * (1.0 - 0.125) * (1.0 - 0.50)

    assert math.isclose(load1, 87.5, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(load2, expected, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(load2, 43.75, rel_tol=0.0, abs_tol=1e-12)


def test_order_of_serial_application_matches_multiplicative_rule():
    load0 = 100.0

    eff_a = 0.30
    eff_b = 0.20

    load_ab = apply_efficiency(apply_efficiency(load0, 1.0, eff_a), 1.0, eff_b)
    load_ba = apply_efficiency(apply_efficiency(load0, 1.0, eff_b), 1.0, eff_a)

    # For the simple multiplicative full-treatment case, order should match.
    assert math.isclose(load_ab, load_ba, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(load_ab, 56.0, rel_tol=0.0, abs_tol=1e-12)