import numpy as np
import pytest

from pines.adapters.spinnaker1 import SpiNNaker1Mapping, signed_connections
from pines.models import DenseRecurrentSNN
from pines.emulator import ScalarInterpreter
from pines.semantics import ExecutionSemantics, NumericFormat, ResetRule


def toy_model():
    return DenseRecurrentSNN(input_weights=np.array([[2.0, -1.0], [0.5, 2.0]]),
        recurrent_weights=np.array([[0.5, -0.2], [0.3, 0.4]]),
        output_weights=np.array([[1.0, -0.5], [-0.5, 1.0]]),
        threshold=np.array([0.7, 0.8]), tau_mem=np.array([5.0, 4.0]),
        bias=np.array([0.1, -0.1]), reset_value=0.0)


def test_signed_weights_keep_source_destination_orientation():
    assert signed_connections(np.array([[0, -2, 1], [3, 0, -4]]), 1.0) == {
        "excitatory": [(0, 2, 1.0, 1.0), (1, 0, 3.0, 1.0)],
        "inhibitory": [(0, 1, 2.0, 1.0), (1, 2, 4.0, 1.0)]}


def test_euler_leak_transport_matches_real_arithmetic_reset_target():
    model = toy_model()
    events = np.random.default_rng(3).integers(0, 2, (8, 50, 2))
    actual = SpiNNaker1Mapping(integration="match_source_euler").emulate(model, events)
    semantics = ExecutionSemantics(reset_rule=ResetRule.TO_VALUE,
        state_format=NumericFormat("float64"), weight_format=NumericFormat("float64"))
    expected = ScalarInterpreter().run(model, events, semantics)
    np.testing.assert_allclose(actual["membrane"], expected.membrane, atol=1e-14)
    np.testing.assert_array_equal(actual["spikes"], expected.spikes)
    np.testing.assert_allclose(actual["logits"], expected.final_logits, atol=1e-14)


def test_native_mapping_has_unit_resistance_and_no_refractory():
    model = toy_model()
    params = SpiNNaker1Mapping().neuron_parameters(model)
    np.testing.assert_array_equal(params["cm"], params["tau_m"])
    np.testing.assert_array_equal(params["tau_refrac"], 0.0)


def test_euler_mapping_rejects_nonpositive_leak():
    with pytest.raises(ValueError, match="Euler leak"):
        SpiNNaker1Mapping(timestep_ms=5.0, integration="match_source_euler").neuron_parameters(toy_model())


def test_binary_event_requirement():
    with pytest.raises(ValueError, match="binary events"):
        SpiNNaker1Mapping().emulate(toy_model(), np.full((1, 2, 2), 2))
