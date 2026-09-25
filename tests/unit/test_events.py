from xreposkill.events import (classify_command, command_of_step, extract_events,
                               split_compound, strip_wrappers)
from tests.conftest import action_step


def test_strip_wrappers():
    assert strip_wrappers("cd /testbed && PYTHONPATH=. timeout 60 pytest x") == "pytest x"


def test_classify_read_search_edit():
    assert classify_command("cat /testbed/a/b.py") == ("read", "/testbed/a/b.py")
    assert classify_command("grep -rn 'separability' astropy/") == ("search", "separability")
    assert classify_command("sed -i 's/a/b/' astropy/modeling/separable.py") \
        == ("edit", "astropy/modeling/separable.py")
    assert classify_command("cat > repro.py << 'EOF'")[0] == "edit"
    assert classify_command("sed -n '100,140p' a.py")[0] == "read"


def test_classify_tests_python_git_submit_install():
    assert classify_command("python -m pytest astropy/tests/test_sep.py")[0] == "run_test"
    assert classify_command("cd /testbed && pytest -x")[0] == "run_test"
    assert classify_command("python repro.py") == ("run_python", "repro.py")
    assert classify_command("git diff HEAD")[0] == "git"
    assert classify_command("echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat p.txt")[0] == "submit"
    assert classify_command("pip install pyerfa")[0] == "install"
    assert classify_command("cd /testbed") is None


def test_command_of_step():
    assert command_of_step({"kind": "action", "raw": {}}) is None
    assert command_of_step(action_step("ls")) == "ls"


def test_extract_events_orders_and_indexes():
    steps = [{"kind": "observation", "content": "task", "raw": {}},
             action_step("cat a.py"),
             {"kind": "thought", "content": "hmm", "raw": {}},
             action_step("sed -i 's/x/y/' a.py"),
             action_step("pytest tests/test_a.py && git diff")]
    assert [(e.step_idx, e.action_type) for e in extract_events(steps)] == [
        (1, "read"), (3, "edit"), (4, "run_test"), (4, "git")]


def test_extract_events_skips_bad_arguments():
    step = {"kind": "action", "raw": {"tool_calls": [{"function": {
        "name": "bash", "arguments": "not json"}}]}}
    assert extract_events([step]) == []


def test_split_compound_quote_and_heredoc_aware():
    assert split_compound('python -c "a; b; c" && git diff') == ['python -c "a; b; c"', 'git diff']
    assert len(split_compound("cat > f.py << 'EOF'\nx; y\nEOF")) == 1
    assert split_compound("echo 'a && b'; ls") == ["echo 'a && b'", "ls"]
