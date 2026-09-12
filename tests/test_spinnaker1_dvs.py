import numpy as np
import pytest

from pines.adapters.spinnaker1_dvs import (
    aggregate_windows, bias_connections, conv2d_connections,
    dense_connections, split_signed_rows, SpiNNaker1DVSMapping,
)


def apply_rows(rows, values, size):
    output = np.zeros(size)
    np.add.at(output, rows[:,1].astype(int), values[rows[:,0].astype(int)]*rows[:,2])
    return output


@pytest.mark.parametrize("shape,kernel,stride", [((2,9,7),(3,2,5,5),2), ((3,6,8),(4,3,3,3),2)])
def test_convolution_indexing_matches_pytorch(shape, kernel, stride):
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(91)
    weights = rng.normal(size=kernel)
    inputs = rng.normal(size=shape)
    rows, output_shape = conv2d_connections(weights, shape, stride)
    expected = torch.nn.functional.conv2d(torch.tensor(inputs)[None], torch.tensor(weights), stride=stride).numpy()[0]
    np.testing.assert_allclose(apply_rows(rows, inputs.ravel(), expected.size).reshape(output_shape), expected, atol=1e-12)


def test_signed_dense_and_bias_keep_destination_orientation_and_layer_delay():
    weights = np.array([[1,0,-2],[0,-3,4]])
    rows = dense_connections(weights)
    receptors = split_signed_rows(rows)
    x = np.array([2,3,5])
    result = apply_rows(receptors["excitatory"],x,2)-apply_rows(receptors["inhibitory"],x,2)
    np.testing.assert_array_equal(result, weights@x)
    bias = bias_connections([1,-2], (2,3), delay=3)
    np.testing.assert_array_equal(bias[:,2], [1]*6+[-2]*6)
    assert np.all(bias[:,3] == 3)


def test_window_aggregation_is_mean_of_softmax_not_softmax_of_mean():
    torch = pytest.importorskip("torch")
    logits = np.array([[10.,0.],[0.,2.]])
    expected = torch.softmax(torch.tensor(logits)/.5, dim=-1).mean(0).numpy()
    np.testing.assert_allclose(aggregate_windows(logits,.5), expected)
    np.testing.assert_array_equal(aggregate_windows(logits,0), [5.,1.])


def test_mapping_is_not_falsely_described_as_fixed_point_state():
    mapping = SpiNNaker1DVSMapping()
    params = mapping.prepare_parameters({"conv1.weight": np.array([-.001, 3.]), "conv1.bias": np.array([-.001])})
    np.testing.assert_array_equal(params["conv1.weight"], [-1/64,127/64])
    np.testing.assert_array_equal(params["conv1.bias"], [-1/256])
    assert "native device" in mapping.contract()["state"]


def test_full_ideal_network_matches_independent_reset_executor():
    torch = pytest.importorskip("torch")
    from pines.benchmarks.dvs_gesture import DVSGestureTrainConfig, build_dvs_conv_srnn
    from pines.semantics import ExecutionSemantics, NumericFormat, ResetRule
    torch.manual_seed(29)
    config = DVSGestureTrainConfig(conv1_channels=2, conv2_channels=3,
                                   hidden_size=4, tau_mem=4, threshold=.125)
    reference = build_dvs_conv_srnn(15, 15, config).double().eval()
    mapping = SpiNNaker1DVSMapping()
    state = {key:value.detach().numpy() for key,value in reference.state_dict().items()}
    realized = mapping.prepare_parameters(state)
    reference.load_state_dict({key:torch.tensor(value) for key,value in realized.items()})
    events = np.random.default_rng(31).integers(0,2,(4,6,2,15,15)).astype(float)
    semantics = ExecutionSemantics(state_format=NumericFormat("float64"),
        weight_format=NumericFormat("float64"), reset_rule=ResetRule.TO_VALUE)
    expected = reference(torch.tensor(events), semantics).detach().numpy()
    observed = mapping.emulate(state, events, 4, .125, record=True)
    np.testing.assert_array_equal(observed["window_logits"], expected)
    assert observed["hidden"].shape == (4,6,4)
    params = mapping.neuron_parameters(4,.125)
    assert np.exp(-1/params["tau_m"]) == .75


def test_device_graph_keeps_convolutions_and_aligns_each_bias_clock():
    import importlib.util
    import sys
    from pathlib import Path
    from types import SimpleNamespace
    scripts = Path(__file__).resolve().parents[1] / "experiments"
    sys.path.insert(0,str(scripts))
    try:
        spec = importlib.util.spec_from_file_location("dvs_device_runner",scripts/"run_spinnaker1_dvs.py")
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
    finally:
        sys.path.pop(0)
    class Population:
        def __init__(self,size,cell,**kwargs):
            self.size,self.label,self.cell = size,kwargs["label"],cell
        def set_max_atoms_per_core(self,n): self.per_core=n
        def initialize(self,**kwargs): pass
        def record(self,name): pass
    projection = lambda pre,post,rows,**kwargs: SimpleNamespace(pre=pre,post=post,rows=rows,**kwargs)
    sim = SimpleNamespace(Population=Population,IF_curr_delta=lambda **kw:kw,
                          Projection=projection,FromListConnector=lambda x:x,StaticSynapse=lambda:None)
    state = {"conv1.weight":np.ones((2,2,5,5))*.125,"conv1.bias":np.ones(2)*.125,
             "conv2.weight":np.ones((3,2,3,3))*.125,"conv2.bias":np.ones(3)*.125,
             "hidden_input.weight":np.ones((4,108))*.125,"hidden_input.bias":np.ones(4)*.125,
             "recurrent.weight":np.eye(4)*.125,"readout.weight":np.ones((11,4))*.125}
    source,clock = object(),object()
    pops,connections,readout = runner.build_network(sim,source,clock,state,
        dict(tau_mem=4,threshold=1,variant="original"),
        SimpleNamespace(conv_neurons_per_core=128,hidden_neurons_per_core=32,record_layers=True),
        SpiNNaker1DVSMapping())
    assert [p.size for p in pops] == [392,108,4]
    clocks = [c for c in connections if c.pre is clock]
    assert [np.unique(c.rows[:,3]).tolist() for c in clocks] == [[1.],[2.],[3.]]
    assert any(c.pre is pops[0] and c.post is pops[1] for c in connections)
    assert any(c.pre is pops[2] and c.post is pops[2] for c in connections)
    assert readout.shape == (4,11)
