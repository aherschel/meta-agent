"""Outer-loop optimizer — split into focused modules.

    state.py          — LoopState + setup, reproducibility manifest, helpers
    proposer.py       — _run_proposer_cli, invoke_proposer, surface-lock
    validate.py       — validate_config, validate_skill
    skill_evolver.py  — invoke_skill_evolver + skill version history
    epoch.py          — _run_one_epoch, baseline + history glue, run_evaluation
    cli.py            — argparse, parse_args, run(args), main()

Public surface (what other parts of meta_agent import):
    run, main, parse_args, build_arg_parser    — from cli
    invoke_proposer                             — from proposer (used by `propose.py`)
    validate_config, validate_skill             — from validate
    invoke_skill_evolver                        — from skill_evolver
"""
from meta_agent.loop.cli import build_arg_parser, main, parse_args, run
from meta_agent.loop.proposer import invoke_proposer
from meta_agent.loop.skill_evolver import invoke_skill_evolver
from meta_agent.loop.validate import validate_config, validate_skill

__all__ = [
    "build_arg_parser",
    "invoke_proposer",
    "invoke_skill_evolver",
    "main",
    "parse_args",
    "run",
    "validate_config",
    "validate_skill",
]
