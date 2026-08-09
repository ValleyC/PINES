from pines.benchmarks.semantic_matrix import primary_semantic_conditions


def test_primary_semantic_conditions_are_unique_and_preregistered() -> None:
    conditions = primary_semantic_conditions()
    assert set(conditions) == {
        "reference",
        "reset_to_value",
        "exponential_euler",
        "pre_integration_threshold",
        "fixed_q8_weights_q16_state",
        "floor_rounding_saturation",
        "synaptic_delay_1",
        "reset_to_value__delay_1",
        "exponential__fixed",
        "pre_threshold__floor",
        "reset__fixed__delay_1",
    }
    hashes = [semantics.semantics_hash for semantics in conditions.values()]
    assert len(hashes) == len(set(hashes))
