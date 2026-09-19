import importlib.util
from pathlib import Path

import pytest

from pines.statistics import clopper_pearson_upper

path=Path(__file__).resolve().parents[1]/"experiments/summarize_spinnaker1_retained.py"
spec=importlib.util.spec_from_file_location("retained",path)
retained=importlib.util.module_from_spec(spec)
spec.loader.exec_module(retained)


def test_complete_capture_recovers_original_bound():
    assert retained.unresolved_upper(7,100,100,.000625)==clopper_pearson_upper(7,100,.000625)


def test_no_capture_cannot_certify_conformance():
    assert retained.unresolved_upper(0,0,160,.000625)==1


def test_all_possible_missing_results_are_covered():
    for missing_errors in range(11):
        assert clopper_pearson_upper(3+missing_errors,30,.05)<=retained.unresolved_upper(3,20,30,.05)


def test_observation_count_does_not_pool_seeds():
    assert retained.unresolved_upper(18,100,861,.000625)>0.9


def test_invalid_counts_raise():
    with pytest.raises(ValueError):
        retained.unresolved_upper(4,3,10,.05)
