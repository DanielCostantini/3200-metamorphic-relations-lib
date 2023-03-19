import pytest

from metamorphic_relations import Info


def test_incorrect_info():
    with pytest.raises(Exception):
        Info([2, 3], [4], [0.5], [0.1])
