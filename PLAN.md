# LabRat build plan

Build one phase at a time. Each phase ends with something you can run and see
working, and builds only on finished phases. Phases 1–4 involve no LLM, so by
the time the agent arrives, everything under it is already tested.

Status: ✅ done · 🔨 in progress · ⬜ not started

---

## ✅ Phase 0: Setup

Python 3.12 env (uv), dependencies, API key in `.env`, folder skeleton.

## ⬜ Phase 1: Train one policy by hand

- **Files:** `labrat/train.py`, `configs/reach.yaml`
- **Build:** FetchReach + SAC + HER, hard-coded. Train, save the model, evaluate
  success rate over 100 episodes, record a rollout video.
- **Done when:** a rollout video shows the arm reliably reaching the target, and
  success rate is above 90%.
- **Learn:** what RL training looks like: reward curves, run time, convergence.
- **Time:** 1–2 days

## ⬜ Phase 2: Make training configurable

- **Files:** `labrat/train.py`, `configs/push.yaml`
- **Build:** everything comes from a config (task, algorithm, learning rate,
  timesteps, seed). Each run writes `runs/<id>/` with `config.yaml`,
  `metrics.json`, videos and the model. W&B logging.
- **Done when:** `uv run python -m labrat.train configs/push.yaml` produces a
  complete run folder, and changing one config value changes the result.
- **Expect:** push with defaults scores low. That's the problem the agent solves later.
- **Time:** 2–3 days

## ⬜ Phase 3: Runner

- **Files:** `labrat/runner.py`
- **Build:** launch training as subprocesses, track status under `runs/`, run
  several in parallel, enforce timeouts and a run budget.
- **Done when:** 4 runs launch at once, one deliberately broken; the other 3
  finish and the broken one is marked `failed` without affecting the others.
- **Time:** 1–2 days

## ⬜ Phase 4: Tools

- **Files:** `labrat/tools.py`
- **Build:** `launch_run`, `get_metrics`, `get_video`, `write_note`,
  `describe_env`, as thin wrappers over the runner that return short summaries.
- **Done when:** each tool, called by hand, returns output you'd be happy to
  hand an LLM.
- **Time:** 1–2 days

## ⬜ Phase 5: Agent loop (hyperparameters only)

- **Files:** `labrat/agent.py`
- **Build:** Claude with tool calling, a goal, a budget and an experiment log.
  The agent changes hyperparameters only.
- **Done when:** started on FetchReach, the agent reaches the goal unattended
  and leaves a readable log of its reasoning.
- **Time:** about a week

## ⬜ Phase 6: Reward editing

- **Files:** `labrat/rewards.py`
- **Build:** the agent can rewrite the reward on FetchPush. A sandbox validates
  agent-written reward code in a separate process before any training.
- **Done when:** the agent improves push success through reward changes, and
  broken reward code is rejected without crashing anything.
- **Time:** about a week

## ⬜ Phase 7: Vision feedback (optional)

- **Files:** `labrat/tools.py` (`get_video`), `labrat/agent.py`
- **Build:** send rollout frames to Claude so it can see failures such as overshooting.
- **Done when:** the agent's notes mention things only visible in the frames.
- **Time:** 3–5 days

## ⬜ Phase 8: Benchmark

- **Files:** `labrat/baselines.py`, `labrat/bench.py`
- **Build:** random search and Optuna through the same runner and budget,
  5 seeds each. Ablations: no vision, no memory, no task card.
- **Done when:** a chart shows runs needed to hit the target per method, with
  error bars.
- **Time:** 3–5 days

## ⬜ Phase 9: Package and post

README, demo video (early runs flailing → agent notes → success), benchmark
chart, and a write-up that includes what failed.

---

**If time runs short:** stop after Phase 6 for a strong demo. Phases 7–8 turn
it into a credible research claim.
