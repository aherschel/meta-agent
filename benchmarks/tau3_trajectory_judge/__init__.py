"""τ³-airline trajectory-judge benchmark package.

Stage 1 of Experiment A (see docs/internal/EXPERIMENT_A_TASK_SUCCESS_PLAN.md):
harness-tune a frozen Claude Haiku 4.5 pairwise task-success judge on a pool
of τ²-labeled airline trajectories.

Modules:
    smoke        — integration smoke-check for tau2.run (R7 de-risk gate)
    build_pool   — (planned) multi-actor pool builder via tau2.run.run_tasks
    adapter      — (planned) meta-agent benchmark adapter; reuses the shared
                    run_judge_benchmark pairwise driver
"""
