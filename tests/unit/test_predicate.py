import pytest

from xreposkill.events import extract_events
from xreposkill.predicate import (PredicateError, evaluate, occurrence_rate,
                                  parse_predicate, validate_predicate)
from tests.conftest import action_step

STEPS = [
    {"kind": "observation", "content": "pr", "raw": {}},
    action_step("grep -rn 'separability_matrix' astropy/"),
    action_step("cat astropy/modeling/separable.py"),
    action_step("python -m pytest astropy/modeling/tests/test_separable.py"),
    action_step("sed -i 's/cright/cleft/' astropy/modeling/separable.py"),
    action_step("python -m pytest astropy/modeling/tests/test_separable.py"),
]


def test_exists():
    assert evaluate('exists(search("separability"))', STEPS)
    assert not evaluate("exists(install)", STEPS)


def test_before_after_first_edit():
    assert evaluate('before(run_test("test_separable"), first_edit)', STEPS)
    assert evaluate("after(run_test, first_edit)", STEPS)
    assert not evaluate("before(first_edit, search)", STEPS)


def test_count_and_logic():
    assert evaluate("count(run_test) >= 2", STEPS)
    assert not evaluate("count(run_test) > 2", STEPS)
    assert evaluate("count(edit) == 1", STEPS)
    assert evaluate("exists(read) and before(search, first_edit)", STEPS)
    assert evaluate("not exists(install)", STEPS)
    assert evaluate("(exists(install) or exists(read)) and exists(git)", STEPS) is False


def test_missing_events_are_false_not_error():
    assert evaluate("before(install, first_edit)", STEPS) is False
    assert evaluate("before(run_test, first_edit)", []) is False


def test_quoted_regex_pattern():
    assert evaluate('exists(run_test("test_separable\\\\.py"))', STEPS)


def test_parse_errors():
    for bad in ("", "exists(", "frobnicate(read)", "count(read) >= x",
                "exists(unknown_type)", 'exists(read("["))',
                "exists(before(run_test, first_edit))", "search('x')"):
        assert validate_predicate(bad) is not None
    with pytest.raises(PredicateError):
        parse_predicate("exists(read) exists(git)")


def test_occurrence_rate():
    evs = [extract_events(STEPS), extract_events([])]
    assert occurrence_rate("exists(edit)", evs) == 0.5
    assert occurrence_rate("exists(edit)", []) == 0.0
