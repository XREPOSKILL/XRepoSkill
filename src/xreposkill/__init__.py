"""XRepoSkill: distil transferable rules from cross-model agent trajectories.

Pipeline stages (each is a subcommand of ``python -m xreposkill``):

    download   fetch mini-SWE-agent trajectories and pass/fail labels
    ingest     normalise every trajectory into the trajectory pool
    pair       pair a failed with a successful trajectory on the same issue
    fork       locate the divergence point of each pair
    discover   Stage 1: candidate rules from three prompts per pair
    merge      Stage 2a: merge candidate rules within each repository
    judge      Stage 2b: coarse LLM screen of merged rules
    score      Stage 2c: resolution gain and z score on held-out trajectories
    audit      Stage 2d: n-gram overlap audit against source patches
    generalize Stage 2e: cross-repository clustering into transferable rules
    pack       write the skill as SKILL.md files
    retrieve   Stage 3a: two-call rule selection per issue
    rollout    Stage 3b: run mini-SWE-agent with the selected rules
"""
__version__ = "1.0.0"
