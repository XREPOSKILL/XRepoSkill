"""Action abstraction: each bash command of a trajectory becomes one
`(action_type, target)` event.

The predicate executor (predicate.py), divergence localization (fork.py),
and the cross-repository clustering (generalize.py) all consume this
abstraction, so the classification lives in one place.

Action types:
    read       cat/head/tail/less/more/nl/sed -n on a file
    search     grep/rg/find/ls/which/locate
    edit       sed -i, redirection into a source file, patch
    run_test   pytest/unittest/tox/runtests invocations
    run_python python -c / python script (non-test)
    install    pip/conda/apt install
    git        git subcommands
    submit     the mini-swe final-output marker
    other      anything else (incl. unparseable tool_calls)
"""
from __future__ import annotations
import json
import re
from dataclasses import dataclass


ACTION_TYPES = ("read", "search", "edit", "run_test", "run_python",
                "install", "git", "submit", "other")


@dataclass(frozen=True)
class Event:
    step_idx: int          # index into the trajectory's steps list
    action_type: str       # one of ACTION_TYPES
    target: str            # compact anchor (path / pattern / test id)
    command: str           # the full bash command


_ENV_PREFIX = re.compile(r"^(?:[A-Za-z_][A-Za-z0-9_]*=\S+\s+)+")
_CD_PREFIX = re.compile(r"^cd\s+\S+\s*&&\s*")
_TIMEOUT_PREFIX = re.compile(r"^timeout\s+(?:-\S+\s+)*\d+\S*\s+")

_SUBMIT_MARKER = re.compile(
    r"COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT|MINI_SWE_AGENT_FINAL_OUTPUT")
_TEST_RUNNER = re.compile(
    r"\b(?:pytest|py\.test|tox)\b|python3?\s+(?:-\S+\s+)*-m\s+(?:pytest|unittest)\b"
    r"|\bruntests?\.py\b")
_SED_INPLACE = re.compile(r"\bsed\b[^|;&]*\s-[a-zA-Z]*i")
_EDIT_REDIRECT = re.compile(
    r"^(?:cat|echo|printf|tee)\b.*?>>?\s*['\"]?/?\S+\.(?:py|pyx|pyi|c|h|cpp"
    r"|cfg|ini|txt|rst|json|yaml|yml|toml)\b")
_PATCH_CMD = re.compile(r"^(?:patch|git\s+apply)\b")
_PATH_TOKEN = re.compile(r"/?[\w./-]+\.[A-Za-z]\w*")
_QUOTED = re.compile(r"""['"]([^'"]+)['"]""")
_TEST_TOKEN = re.compile(r"\S*test\S*")


def strip_wrappers(cmd: str) -> str:
    """Remove cd/env/timeout prefixes so the real program leads."""
    prev = None
    cmd = cmd.strip()
    while prev != cmd:
        prev = cmd
        for pat in (_CD_PREFIX, _ENV_PREFIX, _TIMEOUT_PREFIX):
            cmd = pat.sub("", cmd).strip()
    return cmd


def _first_path(text: str) -> str:
    m = _PATH_TOKEN.search(text)
    return m.group() if m else ""


def _search_target(core: str) -> str:
    m = _QUOTED.search(core)
    if m:
        return m.group(1)
    toks = [t for t in core.split()[1:] if not t.startswith("-")]
    return toks[0] if toks else ""


def classify_command(cmd: str) -> tuple[str, str] | None:
    """Return (action_type, target) for one bash command.

    None for pure shell bookkeeping (bare cd/pwd/export) that carries no
    signal of its own.
    """
    core = strip_wrappers(cmd)
    if not core:
        return None
    prog0 = core.split()[0].rsplit("/", 1)[-1]
    if prog0 in ("cd", "pwd", "export", "true", "set"):
        return None
    if _SUBMIT_MARKER.search(core):
        return "submit", ""
    if prog0 in ("pip", "pip3", "conda", "apt", "apt-get"):
        return "install", _search_target(core)
    if _TEST_RUNNER.search(core):
        m = _TEST_TOKEN.search(core.split(None, 1)[1] if " " in core else "")
        return "run_test", (m.group().strip("'\"") if m else "")
    if _SED_INPLACE.search(core):
        return "edit", _first_path(core)
    if _EDIT_REDIRECT.search(core):
        m = re.search(r">>?\s*['\"]?(/?\S+?\.\w+)", core)
        return "edit", (m.group(1) if m else _first_path(core))
    if _PATCH_CMD.search(core):
        return "edit", _first_path(core)
    prog = core.split()[0].rsplit("/", 1)[-1]
    if prog in ("python", "python2", "python3", "ipython"):
        return "run_python", _first_path(core)
    if prog in ("cat", "head", "tail", "less", "more", "nl", "sed", "awk"):
        return "read", _first_path(core)
    if prog in ("grep", "egrep", "fgrep", "rg", "ack", "find", "ls",
                "which", "locate", "tree"):
        return "search", _search_target(core)
    if prog == "git":
        toks = core.split()
        return "git", (toks[1] if len(toks) > 1 else "")
    return "other", ""


def command_of_step(step: dict) -> str | None:
    """Extract the bash command from an action step's raw payload."""
    raw = step.get("raw") or {}
    for tc in raw.get("tool_calls") or []:
        try:
            args = json.loads(tc["function"]["arguments"])
        except (KeyError, TypeError, json.JSONDecodeError):
            continue
        if isinstance(args, dict) and args.get("command"):
            return str(args["command"])
    return None


def split_compound(cmd: str) -> list[str]:
    """Split on top-level `&&` / `;`, respecting quotes; heredocs unsplit."""
    if "<<" in cmd or len(cmd) > 2000:
        return [cmd]
    parts, buf, quote, i = [], [], "", 0
    while i < len(cmd):
        ch = cmd[i]
        if quote:
            if ch == "\\" and quote == '"':
                buf.append(cmd[i:i + 2])
                i += 2
                continue
            if ch == quote:
                quote = ""
            buf.append(ch)
        elif ch in "'\"":
            quote = ch
            buf.append(ch)
        elif ch == "&" and cmd[i:i + 2] == "&&":
            parts.append("".join(buf))
            buf = []
            i += 2
            continue
        elif ch == ";":
            parts.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
        i += 1
    parts.append("".join(buf))
    return [p.strip() for p in parts if p.strip()]


def extract_events(steps: list[dict]) -> list[Event]:
    """Abstract a trajectory's steps into its ordered Event sequence."""
    events: list[Event] = []
    for i, s in enumerate(steps):
        if s.get("kind") != "action":
            continue
        cmd = command_of_step(s)
        if cmd is None:
            continue
        seen: set[tuple[str, str]] = set()
        for part in split_compound(cmd):
            res = classify_command(part)
            if res is None or res in seen:
                continue
            seen.add(res)
            events.append(Event(step_idx=i, action_type=res[0],
                                target=res[1], command=cmd))
    return events
