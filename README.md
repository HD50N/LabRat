# LabRat

An AI agent that runs robot learning experiments on its own.

You give it a goal, such as "push the block to the target with 90% success,"
and a compute budget. LabRat then works like a junior researcher: it launches
reinforcement learning runs on a simulated robot arm, reads the metrics,
watches rollout videos, writes notes, and decides what to try next, repeating
until the robot hits the goal or the budget runs out.

Everything runs in simulation with MuJoCo, so no physical robot is needed and
a laptop CPU is enough.

> **Status:** early development. A single training run works end to end
> (FetchReach: 100% success in under 2 minutes). Configurable training
> (Phase 2) is next. See [plan.md](plan.md) for the full roadmap.

## Run one training experiment

```bash
uv run python -m labrat.train configs/reach.yaml
```

This trains SAC + HER, evaluates on 100 fresh episodes, and writes
`runs/<timestamp>-reach/` containing `config.yaml`, `metrics.json`,
`monitor.csv` (reward curve), `model.zip`, and `rollout.mp4`.

Watch training curves live while it runs (open http://localhost:6006):

```bash
uv run tensorboard --logdir runs
```

Watch a trained policy live in a MuJoCo window:

```bash
uv run python -m labrat.watch runs/<run_id>
```

## How it will work

```
LLM agent (Claude)
   │  decides the next experiment: hyperparameters, reward function
   ▼
Training run (Stable-Baselines3 SAC + HER, in a subprocess)
   │  produces a trained policy, success rate, rollout video
   ▼
Agent reads the results, logs what it learned, tries again
```

The agent is never trained itself. It's prompted with the goal, a description
of the robot, and its experiment log. Each experiment trains a new robot
policy from scratch.

The headline test: does the agent reach the target success rate in fewer runs
than random search and Optuna under the same budget?

## Setup

Requires macOS or Linux and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/HD50N/LabRat.git
cd LabRat
uv sync
cp .env.example .env   # then add your ANTHROPIC_API_KEY
```

uv installs Python 3.12 and all dependencies into `.venv`.

## Look at the simulation

Open the push scene in MuJoCo's interactive viewer. Double-click an object,
then Ctrl + right-drag to push it around:

```bash
uv run python -m mujoco.viewer --mjcf=.venv/lib/python3.12/site-packages/gymnasium_robotics/envs/assets/fetch/push.xml
```

## Tasks

| Task | Goal | Role |
|---|---|---|
| `FetchReach-v4` | Move the gripper to a target | Pipeline check; trains in minutes |
| `FetchPush-v4` | Push the block onto the target | The real challenge; needs reward design |

Both come from [Gymnasium-Robotics](https://robotics.farama.org/).

## Project layout

```
configs/          one YAML per task
labrat/
  train.py        one training run: env, train, evaluate, record
  watch.py        play a trained policy live in a MuJoCo window
  runner.py       runs training as subprocesses, tracks runs, enforces budget
  tools.py        functions the agent can call
  agent.py        the LLM loop and experiment log
  rewards.py      reward functions + sandbox for agent-written rewards
  baselines.py    random search and Optuna
  bench.py        agent vs. baselines, ablations
runs/             output of every run (git-ignored)
```

Each layer only calls the one below it: train ← runner ← tools ← agent.
[AGENTS.md](AGENTS.md) explains the design rules and known gotchas.

## Stack

MuJoCo · Gymnasium-Robotics · Stable-Baselines3 · PyTorch · Optuna ·
Weights & Biases · Claude API
