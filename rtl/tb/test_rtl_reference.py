from pines.rtl_reference import RTLNeuronConfig, rtl_lif_step, signed_clip_or_wrap


def test_signed_saturation_and_wrap():
    assert signed_clip_or_wrap(200, 8, True) == 127
    assert signed_clip_or_wrap(-200, 8, True) == -128
    assert signed_clip_or_wrap(128, 8, False) == -128


def test_reset_modes_diverge_on_overshoot():
    subtractive = RTLNeuronConfig(state_bits=16, tau_steps=1, reset_mode="subtractive")
    to_value = RTLNeuronConfig(state_bits=16, tau_steps=1, reset_mode="to_value")
    assert rtl_lif_step(0, 300, 192, 0, subtractive) == (108, 1)
    assert rtl_lif_step(0, 300, 192, 0, to_value) == (0, 1)

