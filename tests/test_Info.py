import pytest

from metamorphic_relations.Info import Info

info = Info([2, 3], [4, 5], [0.5, 0.6], [0.1, 0.2])


def test_Info_init_incorrect_lengths():
    with pytest.raises(Exception):
        Info([2, 3], [4], [0.5], [0.1])


def test_Info_to_JSON_correct():
    assert info.to_JSON() == ('{"original_count": [2, 3], "actual_count": [4, 5], "train_f1": [0.5, 0.6], '
                              '"test_f1": [0.1, 0.2], "name": null}')


def test_Info_from_JSON_correct():
    assert Info.from_JSON('{"original_count": [2, 3], "actual_count": [4, 5], "train_f1": [0.5, 0.6], '
                          '"test_f1": [0.1, 0.2], "name": null}').original_count == [2, 3]
    assert Info.from_JSON('{"original_count": [2, 3], "actual_count": [4, 5], "train_f1": [0.5, 0.6], '
                          '"test_f1": [0.1, 0.2], "name": null}').name is None


def test_Info_list_to_info_correct():
    ls = [(1, 2, 0.1, 0.2), (3, 4, 0.3, 0.4)]
    info = Info.list_to_info(ls)
    assert info.original_count == [1, 3]
    assert info.actual_count == [2, 4]
    assert info.train_f1 == [0.1, 0.3]
    assert info.test_f1 == [0.2, 0.4]


def test_Info_list_to_info_incorrect_list_shape():
    ls = [(1, 2, 0.1), (3, 4, 0.3, 0.4)]
    with pytest.raises(Exception):
        Info.list_to_info(ls)


def test_Info_set_name_correct():
    info.set_name("DSMR Results")
    assert info.name == "DSMR Results"
