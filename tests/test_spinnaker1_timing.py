from pines.adapters.spinnaker1_timing import align_run_steps


def test_padding_retains_the_full_observation_and_restores_counter_phase():
    for bits in range(5):
        for minimum in range(1, 129):
            duration = align_run_steps(minimum, bits)
            assert minimum <= duration < minimum + (1 << bits)
            assert duration % (1 << bits) == 0


def test_paper_run_lengths_and_already_aligned_duration():
    assert align_run_steps(67, 4) == 80
    assert align_run_steps(55, 4) == 64
    assert align_run_steps(80, 4) == 80
    assert align_run_steps(67, 0) == 67
