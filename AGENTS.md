# AGENTS.md

Guidance for coding agents working in this repo.

## What this is

LabRat is an AI agent that runs robot learning experiments on its own: it
launches RL training on a simulated robot arm, reads the metrics and rollout
videos, changes what isn't working, and repeats until the robot hits a goal
or the budget runs out. Everything runs in simulation (MuJoCo).

The build is phased. **Read `plan.md` first** and work only on the current
phase. Don't build ahead.

## Working with the owner

- The owner is building this to learn. When you make a design decision,
  explain what you chose, the alternative, and why.
- Keep it lean. Create a file when there is code to put in it; split it only
  when it passes ~300 lines or does two unrelated jobs. No empty stubs for
  future phases.

## Layout

```
configs/          one YAML per task (reach.yaml, push.yaml)
labrat/
  train.py        layer 1: one training run (env, train, evaluate, record)
  watch.py        play a trained policy live (human use only, not a tool)
  runner.py       layer 2: training runs as subprocesses, run registry, budget
  tools.py        layer 3: functions the agent can call
  agent.py        layer 4: LLM loop, provider adapter, experiment log
  rewards.py      reward functions + sandbox for agent-written rewards
  baselines.py    random search and Optuna
  bench.py        agent vs. baselines, ablations
runs/             per-run output, git-ignored
```

## Architecture rules

- **Each layer depends only on the one below it:**
  train ← runner ← tools ← agent. `train.py` must never import the LLM, and
  the agent never calls Stable-Baselines3 directly.
- **Training always runs in a subprocess** (via `runner.py`), never inside the
  agent's process. MuJoCo can crash at the C level.
- **Every run is a folder:** `runs/<id>/config.yaml, metrics.json, *.mp4,
  model.zip, notes.md`. These files are the record. W&B is a dashboard only.
- **Experiments are defined by config, not code.** The one exception is reward
  functions, which always go through the sandbox in `rewards.py` first.
- **Tools return short summaries**, never raw logs or full videos.
- **Baselines use the same runner and budget as the agent**, or the benchmark
  isn't fair.

## Commands

```bash
uv sync                                    # install dependencies
uv run python -m labrat.train configs/reach.yaml   # one training run
uv run python -m labrat.watch runs/<run_id>        # watch a trained policy live
uv run tensorboard --logdir runs                   # live training curves
uv run python -m mujoco.viewer --mjcf=.venv/lib/python3.12/site-packages/gymnasium_robotics/envs/assets/fetch/push.xml
```

Use `uv add <pkg>` to add dependencies, never `pip install`.

## Environment and gotchas

- **Python 3.12**, pinned in `.python-version`. Newer versions lack wheels for
  some robotics packages.
- **Tasks are Gymnasium-Robotics Fetch envs** (`FetchReach-v4`, `FetchPush-v4`),
  not panda-gym. panda-gym depends on PyBullet, which fails to compile on
  Apple Silicon with the current macOS SDK.
- **MuJoCo is pinned to `<3.12`.** 3.12+ breaks a joint-type check in
  gymnasium-robotics (`int in (enum, enum)` returns False). Don't upgrade it
  until gymnasium-robotics fixes this.
- Call `gym.register_envs(gymnasium_robotics)` before `gym.make("Fetch...")`.
- Fetch observations are dicts (`observation`, `achieved_goal`,
  `desired_goal`), so use SB3's `MultiInputPolicy` with `HerReplayBuffer`.
- Training is CPU-only and fine on a laptop: roughly 280 SAC+HER steps/sec,
  so 200k steps takes about 12 minutes.

## Secrets

- The Anthropic key is `ANTHROPIC_API_KEY` in `.env`, loaded with
  `python-dotenv`. `.env` is git-ignored; `.env.example` is the template.
- Never print, log or commit the key, and never write it into `runs/`.
