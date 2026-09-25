"""Pydantic records shared by every stage."""
from __future__ import annotations
import re
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

CATEGORIES = ("lookup", "verification", "replay", "search_recipe")
Category = Literal["lookup", "verification", "replay", "search_recipe"]
RULE_KINDS = ("action", "fact")
RuleKind = Literal["action", "fact"]
ANALYST_TYPES = ("skipped_step", "discipline_observer", "process_delta")
AnalystType = Literal["skipped_step", "discipline_observer", "process_delta"]


def repo_from_instance_id(instance_id: str) -> str:
    """SWE-bench convention: ``<org>__<name>-<num>`` becomes ``<org>/<name>``."""
    org_name = instance_id.rsplit("-", 1)[0]
    org, _, name = org_name.partition("__")
    return f"{org}/{name}"


def repo_slug_from_instance_id(instance_id: str) -> str:
    """``<org>__<name>-<num>`` becomes the bucket name ``<org>__<name>``."""
    return instance_id.rsplit("-", 1)[0]


def repo_slug_from_pair_id(pair_id: str) -> str:
    """pair_ids start with the instance id: ``<iid>::<fail>::<pass>``."""
    return repo_slug_from_instance_id(pair_id.split("::", 1)[0])


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


class TrajStep(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["thought", "action", "observation"]
    content: str
    raw: dict


class NormalizedTraj(BaseModel):
    model_config = ConfigDict(extra="forbid")
    instance_id: str
    repo: str
    agent: str
    model: str
    outcome: Literal["pass", "fail", "unknown"]
    steps: list[TrajStep]
    submission_id: str
    source_path: str
    traj_id: str = ""

    @model_validator(mode="after")
    def _default_traj_id(self) -> "NormalizedTraj":
        if not self.traj_id:
            self.traj_id = f"{self.submission_id}::{self.instance_id}"
        return self


class Evidence(BaseModel):
    model_config = ConfigDict(extra="forbid")
    fail_traj_id: Optional[str] = None
    pass_traj_id: Optional[str] = None
    skipped_step_index: Optional[int] = None
    cited_pass_step_index: Optional[int] = None
    gold_patch_hunk_ref: Optional[str] = None

    @field_validator("skipped_step_index", "cited_pass_step_index",
                     mode="before")
    @classmethod
    def _first_step_index(cls, v):
        # The LLM sometimes cites several steps ("4,6,10" or "5-6");
        # keep the first index instead of rejecting the candidate.
        if isinstance(v, str):
            m = re.search(r"\d+", v)
            if m:
                return int(m.group())
        return v


class Candidate(BaseModel):
    """One candidate rule written by a Stage 1 prompt for one pair."""
    model_config = ConfigDict(extra="forbid")
    pair_id: str
    analyst_type: AnalystType
    kind: RuleKind
    category: Category
    rule: str = Field(min_length=1)
    evidence: Evidence
    predicate: Optional[str] = None     # executable predicate (DSL source)

    @model_validator(mode="after")
    def _evidence_matrix(self) -> "Candidate":
        e = self.evidence
        if self.analyst_type == "skipped_step":
            req = [e.fail_traj_id, e.skipped_step_index, e.gold_patch_hunk_ref]
            if any(v is None for v in req):
                raise ValueError("skipped_step requires fail_traj_id + "
                                 "skipped_step_index + gold_patch_hunk_ref")
        elif self.analyst_type == "discipline_observer":
            if e.pass_traj_id is None:
                raise ValueError("discipline_observer requires pass_traj_id")
        elif self.analyst_type == "process_delta":
            req = [e.fail_traj_id, e.pass_traj_id,
                   e.skipped_step_index, e.cited_pass_step_index]
            if any(v is None for v in req):
                raise ValueError("process_delta requires fail_traj_id + "
                                 "pass_traj_id + skipped_step_index + "
                                 "cited_pass_step_index")
        return self


class Rule(BaseModel):
    """A merged rule of one repository (Stage 2a onward)."""
    model_config = ConfigDict(extra="forbid")
    rule_id: str
    title: str = Field(min_length=1, max_length=120)
    body: str = Field(min_length=1, max_length=600)
    category: Category
    support: int = Field(ge=0)
    source_pair_ids: list[str]
    predicate: Optional[str] = None     # most common predicate of its candidates
    validation: Optional[dict] = None   # Stage 2c gain / z score record
