import pytest

from metamorphic_relations.Results import Results
from metamorphic_relations.Info import Info

result = Results(Info([1, 2], [1, 2], [0.5, 0.6], [0.1, 0.2]), Info([1, 2], [4, 8], [0.6, 0.7], [0.2, 0.3]),
                 Info([1, 2], [3, 6], [0.7, 0.8], [0.3, 0.4]), Info([1, 2], [6, 12], [0.8, 0.9], [0.4, 0.5]))


def test_Results_graph_incorrect_both_f1s():
    with pytest.raises(Exception):
        result.graph(train_f1s=True, test_f1s=True)


def test_Results_graph_incorrect_neither_f1s():
    with pytest.raises(Exception):
        result.graph(train_f1s=False, test_f1s=False)


def test_Results_graph_incorrect_show_sets():
    with pytest.raises(Exception):
        result.graph(show_sets=False)


def test_Results_get_forall_sets_correct():
    assert result.get_forall_sets(get_set_function=lambda x: x.actual_count) == [[1, 2], [4, 8], [3, 6], [6, 12]]


def test_Results_get_forall_sets_correct_subsets():
    assert result.get_forall_sets((True, False, False, True), lambda x: x.train_f1) == [[0.5, 0.6], [0.8, 0.9]]


def test_Results_get_forall_sets_incorrect_is_set():
    with pytest.raises(Exception):
        result.get_forall_sets(False, lambda x: x.train_f1)


def test_Results_write_to_file_correct():
    assert result.write_to_file("Output/test1.txt") == (
        '{"original_results": "{\\"original_count\\": [1, 2], \\"actual_count\\": [1, '
        '2], \\"train_f1\\": [0.5, 0.6], \\"test_f1\\": [0.1, 0.2], \\"name\\": '
        'null}", "GMR_results": "{\\"original_count\\": [1, 2], \\"actual_count\\": '
        '[4, 8], \\"train_f1\\": [0.6, 0.7], \\"test_f1\\": [0.2, 0.3], \\"name\\": '
        'null}", "DSMR_results": "{\\"original_count\\": [1, 2], \\"actual_count\\": '
        '[3, 6], \\"train_f1\\": [0.7, 0.8], \\"test_f1\\": [0.3, 0.4], \\"name\\": '
        'null}", "all_MR_results": "{\\"original_count\\": [1, 2], '
        '\\"actual_count\\": [6, 12], \\"train_f1\\": [0.8, 0.9], \\"test_f1\\": '
        '[0.4, 0.5], \\"name\\": null}", "individual_results": "None"}')


def test_Results_write_to_file_correct_None():
    results1 = Results()
    assert results1.write_to_file("Output/test2.txt") == (
        '{"original_results": "None", "GMR_results": "None", "DSMR_results": "None", '
        '"all_MR_results": "None", "individual_results": "None"}')


def test_Results_read_from_file_correct():
    results1 = Results.read_from_file("Output/test1.txt")

    assert results1.original_results.train_f1 == [0.5, 0.6]
    assert results1.GMR_results.test_f1 == [0.2, 0.3]
    assert results1.DSMR_results.original_count == [1, 2]
    assert results1.all_MR_results.actual_count == [6, 12]
    assert results1.individual_results == [None]


def test_Results_read_from_file_correct_None():
    results1 = Results.read_from_file("Output/test2.txt")

    assert results1.original_results is None
    assert results1.GMR_results is None
    assert results1.DSMR_results is None
    assert results1.all_MR_results is None
    assert results1.individual_results == [None]


def test_Results_get_JSON_correct():
    info = Info([1, 2], [3, 4], [0.1, 0.2], [0.3, 0.4])

    assert Results.get_JSON(info) == ('{"original_count": [1, 2], "actual_count": [3, 4], "train_f1": [0.1, 0.2], '
                                      '"test_f1": [0.3, 0.4], "name": null}')


def test_Results_get_JSON_correct_None():
    assert Results.get_JSON(None) == "None"
