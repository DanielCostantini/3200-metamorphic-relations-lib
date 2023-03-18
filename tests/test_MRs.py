import pytest

from metamorphic_relations import ImageMRs
from metamorphic_relations import Results
from metamorphic_relations import Info


def test_incorrect_info():
    with pytest.raises(Exception):
        Info([2, 3], [4], [0.5], [0.1])

def test_get_forall_sets():

    result = Results(Info([2], [4], [0.5], [0.1]), Info([2], [3], [0.5], [0.1]))

    assert result.get_forall_sets(lambda x: x.actual_count) == [4, 3]

