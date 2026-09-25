"""Executable rule predicates: a small DSL over a trajectory's event sequence.

    expr     := clause (("and" | "or") clause)*
    clause   := ["not"] atom
    atom     := "exists(" event ")"
              | "before(" event "," event ")"
              | "after(" event "," event ")"
              | "count(" event ")" cmp INT
              | "(" expr ")"
    event    := TYPE | TYPE "(" pattern ")" | "first_edit"
    pattern  := "*" | bare-token | quoted regex ("...")
    cmp      := ">=" | ">" | "<=" | "<" | "=="

An event ``TYPE(pattern)`` matches a trajectory event when the action_type
equals TYPE (or TYPE is "any") and the regex ``pattern`` searches the full
command text, case-insensitively.  ``first_edit`` matches only the earliest
edit event.  ``before(E1, E2)``: some E1 match strictly precedes some E2
match; ``after(E1, E2)``: some E1 match strictly follows some E2 match.
Missing events make the containing atom False, never an error.  Evaluation
needs no LLM call.
"""
from __future__ import annotations
import json
import re
from dataclasses import dataclass

from xreposkill.events import ACTION_TYPES, Event, extract_events


class PredicateError(ValueError):
    """Raised when an expression cannot be parsed."""


_TOKEN = re.compile(r"""
    \s*(?:
        (?P<quoted>"(?:[^"\\]|\\.)*")
      | (?P<cmp>>=|<=|==|>|<)
      | (?P<punct>[(),*])
      | (?P<word>[A-Za-z_][A-Za-z0-9_.\-/]*|\d+)
    )""", re.VERBOSE)


def _tokenize(expr: str) -> list[str]:
    toks, pos = [], 0
    while pos < len(expr):
        m = _TOKEN.match(expr, pos)
        if not m or m.end() == pos:
            if expr[pos:].strip():
                raise PredicateError(f"bad token at: {expr[pos:pos+20]!r}")
            break
        toks.append(m.group().strip())
        pos = m.end()
    return toks


@dataclass(frozen=True)
class EventPattern:
    action_type: str            # one of ACTION_TYPES, "any", or "first_edit"
    pattern: str                # regex source; "" means match-all

    def matches(self, ev: Event) -> bool:
        if self.action_type not in ("any", "first_edit") \
                and ev.action_type != self.action_type:
            return False
        if not self.pattern or self.pattern == "*":
            return True
        return bool(re.search(self.pattern, ev.command, re.IGNORECASE))

    def match_indices(self, events: list[Event]) -> list[int]:
        if self.action_type == "first_edit":
            for i, ev in enumerate(events):
                if ev.action_type == "edit":
                    return [i]
            return []
        return [i for i, ev in enumerate(events) if self.matches(ev)]


@dataclass(frozen=True)
class Node:
    op: str                     # exists|before|after|count|and|or|not
    events: tuple[EventPattern, ...] = ()
    children: tuple["Node", ...] = ()
    cmp: str = ""
    n: int = 0


class _Parser:
    def __init__(self, toks: list[str]):
        self.toks = toks
        self.i = 0

    def peek(self) -> str | None:
        return self.toks[self.i] if self.i < len(self.toks) else None

    def take(self, expect: str | None = None) -> str:
        t = self.peek()
        if t is None or (expect is not None and t != expect):
            raise PredicateError(f"expected {expect!r}, got {t!r}")
        self.i += 1
        return t

    def parse(self) -> Node:
        node = self.expr()
        if self.peek() is not None:
            raise PredicateError(f"trailing tokens: {self.toks[self.i:]}")
        return node

    def expr(self) -> Node:
        left = self.clause()
        while self.peek() in ("and", "or"):
            op = self.take()
            right = self.clause()
            left = Node(op=op, children=(left, right))
        return left

    def clause(self) -> Node:
        if self.peek() == "not":
            self.take()
            return Node(op="not", children=(self.clause(),))
        return self.atom()

    def atom(self) -> Node:
        t = self.peek()
        if t == "(":
            self.take("(")
            node = self.expr()
            self.take(")")
            return node
        if t in ("exists", "count"):
            self.take()
            self.take("(")
            ev = self.event()
            self.take(")")
            if t == "exists":
                return Node(op="exists", events=(ev,))
            cmp = self.take()
            if cmp not in (">=", ">", "<=", "<", "=="):
                raise PredicateError(f"bad comparator {cmp!r}")
            n = self.take()
            if not n.isdigit():
                raise PredicateError(f"count needs int, got {n!r}")
            return Node(op="count", events=(ev,), cmp=cmp, n=int(n))
        if t in ("before", "after"):
            self.take()
            self.take("(")
            e1 = self.event()
            self.take(",")
            e2 = self.event()
            self.take(")")
            return Node(op=t, events=(e1, e2))
        raise PredicateError(f"unexpected token {t!r}")

    def event(self) -> EventPattern:
        t = self.take()
        if t == "first_edit":
            return EventPattern("first_edit", "")
        if t == "any" or t in ACTION_TYPES:
            atype = t
        else:
            raise PredicateError(f"unknown action_type {t!r}")
        if self.peek() != "(":
            return EventPattern(atype, "")
        self.take("(")
        p = self.take()
        if p == "*":
            pattern = ""
        elif p.startswith('"'):
            try:
                pattern = json.loads(p)
            except ValueError as e:
                raise PredicateError(f"bad quoted pattern {p!r}: {e}") from e
        else:
            pattern = re.escape(p)
        self.take(")")
        try:
            re.compile(pattern)
        except re.error as e:
            raise PredicateError(f"bad regex {pattern!r}: {e}") from e
        return EventPattern(atype, pattern)


def parse_predicate(expr: str) -> Node:
    toks = _tokenize(expr)
    if not toks:
        raise PredicateError("empty predicate")
    return _Parser(toks).parse()


def validate_predicate(expr: str) -> str | None:
    """``None`` when ``expr`` parses, else the error message."""
    try:
        parse_predicate(expr)
        return None
    except PredicateError as e:
        return str(e)


def eval_node(node: Node, events: list[Event]) -> bool:
    if node.op == "and":
        return all(eval_node(c, events) for c in node.children)
    if node.op == "or":
        return any(eval_node(c, events) for c in node.children)
    if node.op == "not":
        return not eval_node(node.children[0], events)
    if node.op == "exists":
        return bool(node.events[0].match_indices(events))
    if node.op == "count":
        c = len(node.events[0].match_indices(events))
        return {"<": c < node.n, "<=": c <= node.n, ">": c > node.n,
                ">=": c >= node.n, "==": c == node.n}[node.cmp]
    if node.op in ("before", "after"):
        i1 = node.events[0].match_indices(events)
        i2 = node.events[1].match_indices(events)
        if not i1 or not i2:
            return False
        if node.op == "before":
            return min(i1) < max(i2)
        return max(i1) > min(i2)
    raise PredicateError(f"unknown op {node.op}")


def evaluate(expr: str | Node, steps: list[dict]) -> bool:
    """Evaluate a predicate over a trajectory's steps."""
    node = parse_predicate(expr) if isinstance(expr, str) else expr
    return eval_node(node, extract_events(steps))


def occurrence_rate(expr: str, event_lists: list[list[Event]]) -> float:
    """Fraction of trajectories (given as event lists) satisfying ``expr``."""
    if not event_lists:
        return 0.0
    node = parse_predicate(expr)
    return sum(eval_node(node, evs) for evs in event_lists) / len(event_lists)
