import pytest

from metamorphic_relations import Results
from metamorphic_relations import Info

result = Results(Info([1, 2], [1, 2], [0.5, 0.6], [0.1, 0.2]), Info([1, 2], [4, 8], [0.6, 0.7], [0.2, 0.3]),
                 Info([1, 2], [3, 6], [0.7, 0.8], [0.3, 0.4]), Info([1, 2], [6, 12], [0.8, 0.9], [0.4, 0.5]))


def test_get_forall_sets():
    assert result.get_forall_sets(get_set_function=lambda x: x.actual_count) == [[1, 2], [4, 8], [3, 6], [6, 12]]


def test_get_forall_sets_subsets():
    assert result.get_forall_sets([True, False, False, True], lambda x: x.train_f1) == [[0.5, 0.6], [0.8, 0.9]]


def test_write_to_file():
    assert result.write_to_file("Output/test1.txt") == (
        '{"original_results": "{\\"original_count\\": [1, 2], \\"actual_count\\": [1, '
        '2], \\"train_f1\\": [0.5, 0.6], \\"test_f1\\": [0.1, 0.2]}", "GMR_results": '
        '"{\\"original_count\\": [1, 2], \\"actual_count\\": [4, 8], \\"train_f1\\": '
        '[0.6, 0.7], \\"test_f1\\": [0.2, 0.3]}", "DSMR_results": '
        '"{\\"original_count\\": [1, 2], \\"actual_count\\": [3, 6], \\"train_f1\\": '
        '[0.7, 0.8], \\"test_f1\\": [0.3, 0.4]}", "all_MR_results": '
        '"{\\"original_count\\": [1, 2], \\"actual_count\\": [6, 12], \\"train_f1\\": '
        '[0.8, 0.9], \\"test_f1\\": [0.4, 0.5]}"}')


def test_read_from_file():
    assert Results.read_from_file("Output/test1.txt").DSMR_results.test_f1 == [0.3, 0.4]


def test_graph():
    result.graph()


def test_graph_train_f1s():
    result.graph(train_f1s=True, test_f1s=False)


def test_graph_both_f1s():
    with pytest.raises(Exception):
        result.graph(train_f1s=True, test_f1s=True)


def test_graph_neither_f1s():
    with pytest.raises(Exception):
        result.graph(train_f1s=False, test_f1s=False)


def test_graph_incorrect_show_sets():
    with pytest.raises(Exception):
        result.graph(show_sets=["False"])
