# Make Heuristics Great Again: Letting Codex Build Heuristic Systems from Scratch

> Jiayi Weng

I recently ran a set of experiments in EnvPool. I did not train neural networks, and I did not run reinforcement learning. I just gave the environments to Codex and let it write policies, run evaluations, read logs, edit code, and keep pushing Atari and MuJoCo scores higher. The results were more extreme than I expected: Pong reached `21`; Breakout reached `864` with both RAM input and pure image input; in MuJoCo, Ant-v5 reached `6146`, and HalfCheetah-v5 reached `12041`.

This is not meant to be a clickbait claim that "a few if-else statements can beat any reinforcement-learning policy." The part I care about is the cost structure. In the past, heuristic systems were hard to maintain. You had to write detectors, infer action meanings, handle frame skip and serving, maintain conditional branches, run parameter sweeps, and keep track of where every score came from. It is hard for a human to keep all of that in their head over time, so heuristic policies easily turn into one-off scripts.

Codex changed how I feel about that tradeoff. It does not just write a `policy.py` and stop. It keeps doing the things you would do when maintaining a small software project: write tests, run batches of experiments, watch videos, change parameters, add logs, draw sample-efficiency curves, delete bad directions, and freeze good configurations. Heuristics used to feel like an old-fashioned hack. Now I think they may be exactly the kind of thing coding agents are good at maintaining.

I will first describe how the experiments were run, then go through three examples in order. Breakout comes first because it is the most visually intuitive: you can see an Atari policy being tuned step by step. Ant comes next because in continuous control the phenomenon looks more like a reflex system growing on its own. Finally I discuss Atari57, because although it is less clean as a single-task story, it better shows what the distribution looks like when this method runs in a batch without human intervention.

## How the Experiments Ran

For model configuration, these experiments mainly used Codex 5.4 with `xhigh`. I did not run a systematic ablation over model size or reasoning effort here, so this article is about what happened under that configuration, not a comparison across models.

The starting point was actually an engineering problem: I wanted to validate EnvPool environments, and I needed policies that were much stronger than random policies but did not require training. Random policies are too weak. In many environments, an entire episode never reaches the part of the task that gives meaningful reward, so when something fails all you see is a timeout or a score of 0. It is hard to tell whether the environment is wrong, the wrapper is wrong, or the policy simply never reached an informative state. Training a neural network for every environment is too heavy too. Versions, dependencies, training budgets, and checkpoints would all become burdens inside the test system itself.

At first I tried the most direct version: ask Codex to "write a policy that solves Breakout," let it run one episode, look at the score, and keep editing. That worked poorly because a low score was not informative. It did not know whether the action semantics were wrong, the state reader was wrong, the evaluation setup was wrong, or the policy structure itself was wrong. Later I changed the task so that Codex first had to maintain a complete loop: write detectors, then write the policy, then run full episodes, then write the experiment ledger, and only then freeze `policy.py`.

That loop was first shaped on Pong and Breakout. The earliest detectors only checked action space and observation shape. Later they became action-semantics probes, pixel-threshold detectors, RAM-byte-to-image-position correlation probes, short video replays, and dumps of the last few dozen trajectory steps. Only after that did `trials.jsonl` / `summary.csv` become part of the process, with every change recording cumulative environment steps, score, configuration, and failure reason. The version that worked best was: first probe observable state, then propose one small change, then recheck a full episode, and only promote the change after it passes. Every time a new best score appeared, Codex also did a simplification pass to delete temporary scripts and dead branches that grew during search.

The most time-consuming part of developing this loop was making clear what the policy could see, what each action meant, and where a score came from. In Pong, the slow part was thresholding and aligning RAM coordinates. In Breakout, it was telling apart "can catch the ball" from "can keep clearing bricks." In Ant, it was separating real environment steps, MuJoCo model-planning steps, and experiment-ledger steps. The process felt like turning a black box that only returned a score into a debuggable software system.

The final loop was roughly:

1. Write an evaluation harness and confirm observations, actions, rewards, termination, random seeds, and frame-skip settings.
2. Append every experiment to `trials.jsonl`, then generate `summary.csv` with `env_steps`, score, configuration, and notes.
3. Write probe scripts to find controllable state variables, such as RAM bytes, RGB connected components, MuJoCo joint order, root velocity, and contact patterns.
4. Change one direction at a time: detector, target point, stuck handling, planner objective, candidate action library, or short-horizon proxy objective.
5. Promote a configuration to the default only after a full-episode recheck passes.

This may sound like ordinary experiment hygiene, but it mattered a lot for Codex. Without an experiment ledger, after dozens of rounds it easily mixed up "better for 300 steps" with "better for a full episode," and it also mixed up real environment steps with internal rollouts inside a MuJoCo model. With a ledger, it behaved much more like a software maintainer: every change had a source, every failed direction had a note, and the next round was less likely to repeat the same mistake.

## Breakout: From Just Catching the Ball to the Theoretical Maximum Score of 864

Breakout looks like a geometry problem at first: where the ball is, where the paddle is, and where the ball will land after bouncing off walls. The annoying part comes later: a policy can keep catching the ball forever and still stop breaking new bricks.

Codex did not rush to write the final policy in the first round. It first confirmed the action space and observation shape, then found the paddle, ball, and brick colors in RGB frames, and then used image labels to scan the 128 RAM bytes. The early experiment ledger looked roughly like this:

```text
trial_name                 score   cumulative_env_steps   note
shape_action_probe          -      32                     inspect obs/info/action
ram_byte_corr_probe_v1      -      5,032                  correlate RAM bytes
ram_fit_action_probe_v2     -      9,532                  action 2=right, 3=left
baseline_v0                99      16,303                 initial RAM intercept
tunnel0_v1                387      43,303                 no tunnel offset
```

`387` was the first local high score that could easily fool you. The policy could catch the ball reliably, but it was just feeding the ball into a cycle: it would not die, and it would not clear new bricks. If a human were hand-writing the heuristic, it would be easy to think the problem was still "the intercept is not accurate enough" and keep tuning that. Codex figured out from the video and the last few dozen trajectory steps that the real problem was the lack of disturbance in the ball path.

<video controls src="heuristic_breakout_score387_tunnel0_render210x160.mp4" width="360"></video>

In the `387` video, the problem is visible: the ball and paddle have settled into a stable loop, but brick progress has stopped.

<details>
<summary>Reproduce score `387`</summary>

The command below assumes the current directory is this article's artifact directory, meaning the directory containing `heuristic_breakout.py`. It writes the temporary experiment ledger to `/tmp`, so it does not pollute the ledger used in the article.

```bash
rm -f /tmp/repro_breakout_387.jsonl /tmp/repro_breakout_387.csv
python heuristic_breakout.py \
  --policy ram \
  --episodes 1 \
  --seed 0 \
  --max-steps 27000 \
  --deadband 3 \
  --chase-lead-steps 6 \
  --tunnel-offset 0 \
  --launch-offset 24 \
  --fast-ball-min-vy 1000000000 \
  --stuck-trigger-steps 1000000000 \
  --stuck-switch-steps 0 \
  --stuck-offset 0 \
  --stuck-release-horizon-steps 0 \
  --brick-balance-bias-min-score 1000000000 \
  --late-game-paddle-lag-px 0 \
  --trial-name repro_breakout_387 \
  --log-path /tmp/repro_breakout_387.jsonl \
  --summary-path /tmp/repro_breakout_387.csv
```

The expected output should include `score=387.0` and `mean=387.000`.

</details>

The first key mechanism it added was a loop breaker: if there had been no reward for a long time, add a periodic offset to the predicted landing point, so the paddle knocks the ball out of the local cycle. That change pushed the score from `387` to `507`.

```python
if steps_since_reward >= stuck_trigger_steps:
    phase = stuck_offset_index % 4
    if phase == 0:
        offset = +stuck_offset_px
    elif phase == 1:
        offset = -stuck_offset_px
    elif phase == 2:
        offset = +0.5 * stuck_offset_px
    else:
        offset = -0.5 * stuck_offset_px
else:
    offset = 0.0
```

<video controls src="heuristic_breakout_score507_stuckbreaker_render210x160.mp4" width="360"></video>

The important part of `507` is not catch accuracy. It is ball-path control: the policy starts deliberately kicking the ball out of local loops.

<details>
<summary>Reproduce score `507`</summary>

```bash
rm -f /tmp/repro_breakout_507.jsonl /tmp/repro_breakout_507.csv
python heuristic_breakout.py \
  --policy ram \
  --episodes 1 \
  --seed 0 \
  --max-steps 27000 \
  --deadband 3 \
  --chase-lead-steps 6 \
  --tunnel-offset 0 \
  --launch-offset 24 \
  --fast-ball-min-vy 1000000000 \
  --stuck-trigger-steps 1024 \
  --stuck-switch-steps 256 \
  --stuck-offset 12 \
  --stuck-release-horizon-steps 0 \
  --brick-balance-bias-min-score 1000000000 \
  --late-game-paddle-lag-px 0 \
  --trial-name repro_breakout_507 \
  --log-path /tmp/repro_breakout_507.jsonl \
  --summary-path /tmp/repro_breakout_507.csv
```

The expected output should include `score=507.0` and `mean=507.000`.

</details>

Then another failure mode appeared. For fast low balls, chasing the normal intercept made the paddle overreact to the lookahead and drift away. Codex added `fast_low_ball_lead_steps=3`, and the score jumped directly from `507` to `839`.

```python
if vy > 0.1 and ball_y <= paddle_y:
    steps_to_paddle = max((paddle_y - ball_y) / vy, 0.0)
    intercept_x = reflect_position(ball_x + vx * steps_to_paddle)
    target_x = intercept_x + stuck_offset
elif vy >= fast_ball_min_vy:
    target_x = ball_x + fast_low_ball_lead_steps * vx
else:
    target_x = ball_x + chase_lead_steps * vx
```

<video controls src="heuristic_breakout_score839_fastlead_render210x160.mp4" width="360"></video>

The `839` video shows another change: low fast balls no longer drag the paddle off course, and the policy finally enters the stage where it can clear the back half of the wall.

<details>
<summary>Reproduce score `839`</summary>

```bash
rm -f /tmp/repro_breakout_839.jsonl /tmp/repro_breakout_839.csv
python heuristic_breakout.py \
  --policy ram \
  --episodes 1 \
  --seed 0 \
  --max-steps 27000 \
  --deadband 3 \
  --chase-lead-steps 6 \
  --tunnel-offset 0 \
  --launch-offset 24 \
  --fast-ball-min-vy 3 \
  --fast-low-ball-lead-steps 3 \
  --stuck-trigger-steps 1024 \
  --stuck-switch-steps 256 \
  --stuck-offset 12 \
  --stuck-release-horizon-steps 0 \
  --brick-balance-bias-min-score 1000000000 \
  --late-game-paddle-lag-px 0 \
  --trial-name repro_breakout_839 \
  --log-path /tmp/repro_breakout_839.jsonl \
  --summary-path /tmp/repro_breakout_839.csv
```

The expected output should include `score=839.0` and `mean=839.000`.

</details>

The move from `839` to `864` was the part that most looked like maintaining a complicated heuristic system. Codex tried deadbands, serve offsets, stuck offsets, brick-balance bias, and lookahead steps. Many directions did not help. The useful change was a late-stage condition: after the first wall has been cleared, the stuck offset should only apply while the ball is still far from the paddle; as the ball gets close, the offset should decay away, otherwise the final few bricks cause the policy to steer the paddle off course. It also added a tiny paddle-drift compensation for the one-step delay between action and paddle position.

```python
if score >= 432 and stuck_release_horizon_steps > 0:
    release_ratio = clip(steps_to_paddle / stuck_release_horizon_steps, 0.0, 1.0)
    offset *= release_ratio

if score >= 432 and ball_y >= 170 and last_action == RIGHT:
    control_paddle_x = paddle_x + 2.0
elif score >= 432 and ball_y >= 170 and last_action == LEFT:
    control_paddle_x = paddle_x - 2.0
```

<video controls src="heuristic_breakout_ci3985ae2_score864_render210x160.mp4" width="360"></video>

<details>
<summary>Reproduce score `864`</summary>

```bash
rm -f /tmp/repro_breakout_864.jsonl /tmp/repro_breakout_864.csv
python heuristic_breakout.py \
  --policy ram \
  --episodes 1 \
  --seed 0 \
  --max-steps 108000 \
  --deadband 3 \
  --chase-lead-steps 6 \
  --tunnel-offset 0 \
  --launch-offset 24 \
  --fast-ball-min-vy 3 \
  --fast-low-ball-lead-steps 3 \
  --stuck-trigger-steps 1024 \
  --stuck-switch-steps 256 \
  --stuck-offset 12 \
  --stuck-release-horizon-steps 8 \
  --brick-balance-deadzone 0.01 \
  --brick-balance-bias-min-score 432 \
  --late-game-paddle-lag-px 2 \
  --late-game-lag-ball-y 170 \
  --trial-name repro_breakout_864 \
  --log-path /tmp/repro_breakout_864.jsonl \
  --summary-path /tmp/repro_breakout_864.csv
```

The expected output should include `score=864.0` and `mean=864.000`.

</details>

The final RAM default configuration verified as `864/864/864` across three episodes. More interestingly, Codex later migrated the same geometric control back to pure image input: no RAM, only RGB segmentation to find the paddle, ball, and brick balance. The pure-image version first reached `310`, then `428`, and finally reached `864` after lowering the late-stage "decay the stuck offset" threshold so it applied throughout the run. It first hit `864` after 7 policy-local episodes, corresponding to `14,504` policy-local environment steps in the figure.

![Breakout sample efficiency](heuristic_breakout_sample_efficiency.png)

The full policy code is in [`heuristic_breakout.py`](heuristic_breakout.py), and the experiment ledger is in [`heuristic_breakout_trials_summary.csv`](heuristic_breakout_trials_summary.csv).

The important point is not "pure image reaches the maximum score from scratch in `14.5K` steps." That would be inaccurate. More precisely, Codex first found geometric control, loop breaking, and late-stage offset decay in the RAM version. Once that structure stabilized, it swapped the state reader from RAM to an RGB detector. The pure-image `14.5K` is a transfer budget, not a from-scratch budget. It says something else: once a heuristic policy is written as a maintainable software structure, the input layer can be replaced, the control logic can be reused, and the result can keep being regression-tested instead of training a new system for every observation format.

## Ant: Codex Introduced Gait Control and Model Predictive Planning on Its Own

If Breakout's geometry is still fairly intuitive, Ant was more surprising to me. I did not start by saying "use CPG," and I did not say "use MPC"; in fact, I still did not know what MPC was at the time. My request was just the same loop as above: write a heuristic policy, do not train a neural network, improve the score, and leave logs plus reproduction commands. Codex first read the EnvPool/Gymnasium Ant observations and rewards, confirmed the action order, root velocity, torso orientation, joint positions, and joint velocities, and then proposed its first rhythmic gait.

The first version was a four-leg phase oscillator: left and right legs in opposite phase, hip and ankle joints tracking sinusoidal target angles, with actions produced by a PD controller. It was not elegant, but it was already much stronger than random. Across 5 random seeds, the mean score was `2291`.

```python
leg_phase = warp_phase(phase + LEG_PHASE, stance_duty(vx))
stance = leg_phase < pi

hip_wave = HIP_BIAS + stance_or_swing_scale * (
    HIP_AMP * sin(leg_phase)
    + HIP_H2_AMP * sin(2 * leg_phase + HIP_H2_PHASE)
    + HIP_H3_AMP * sin(3 * leg_phase + HIP_H3_PHASE)
)

action[0::2] = KP * (
    HIP_SIGN * hip_wave
    + HEADING_AXIS * (YAW_GAIN * yaw + YAW_RATE_GAIN * yaw_rate)
    - q[0::2]
)
action[1::2] = KP * (ANKLE_SIGN * (ankle_wave + balance) - q[1::2])
```

The early iterations afterward felt a lot like tuning a real robot: add yaw feedback to reach `2718`, then tune phase speed, hip/ankle amplitudes, and yaw-rate gain to reach `3025`, then add second- and third-order harmonics to reach `3162`. Codex also tried broad CEM search, but the results did not consistently beat the current rhythmic policy, so it did not simply keep increasing the search scale. It moved to a different representation.

The jump in Ant came from Codex introducing residual model predictive planning, written as MPC in the code. My understanding is: keep the rhythmic gait as a "reflex," then at every real environment step sample dozens of small residual action sequences inside a local MuJoCo model, score them, execute only the first residual action, and warm-start the remaining plan at the next step. This way, the policy does not need to plan how all 8 joints should move from scratch at every step. It starts from a stable base gait, then uses short-horizon model planning to correct that gait.

You can think of MPC as "walking while thinking a short distance ahead." It is not training a policy network, and it is not computing one complete trajectory from step 0 to the end. At the current state, it uses a model to try several possible short future action sequences, chooses the one that looks best, executes only the first step, then observes the next state and tries again. For Ant, this fits well around a heuristic policy: the rhythmic gait makes the robot behave like something that can walk, while MPC makes small corrections at each step so it falls less, yaws less, and moves forward faster.

```python
base = cpg_action(phase, q, dq, roll, pitch, yaw, rates, contacts, vx)

best_plan = previous_plan.copy()
best_obj = rollout_objective(obs, best_plan)
for _ in range(CANDIDATES - 1):
    residuals = clip(
        best_plan + rng.normal(0.0, MPC_SIGMA, size=(HORIZON, 8)),
        -MPC_CLIP,
        MPC_CLIP,
    )
    residuals[1:] = 0.6 * residuals[1:] + 0.4 * residuals[:-1]
    obj = rollout_objective(obs, residuals)
    if obj > best_obj:
        best_obj = obj
        best_plan = residuals

plan[:-1] = PLAN_DECAY * best_plan[1:]
return clip(base + best_plan[0], -1.0, 1.0)
```

This was the part that surprised me most: I did not tell it that model predictive planning was a good fit for Ant. After the rhythmic policy reached `3162`, it wrote a model-based residual planner on its own. The first version, with horizon 6, 32 candidates, and small residuals, moved the score from `3135` to `3635`. Then it kept maintaining the system:

```text
trial_name                               score_mean   cumulative_env_steps   note
ant_lr_cpgpd_v1                         2291.9       5,000                  left/right anti-phase CPG + PD
ant_yawaxis_grid_v2                     2857.9       20,000                 yaw feedback + retuned params
ant_h3_428_v1                           3162.0       50,000                 second/third harmonics
ant_mpc_residual_v1_ep1                 3635.5       62,000                 horizon=6, candidates=32
ant_mpc_residual_cfg4_eval5             3964.7       67,000                 horizon=8, candidates=48
ant_mpc_residual_cand07_eval5           4647.1       73,000                 local search around MPC config
ant_mpc_residual_narrow04_eval5         4871.3       79,000                 lower z target, higher kp/candidates
ant_mpc_residual_warm02_eval5           5165.2       85,000                 warm-start residual plan
ant_mpc_fast065x060_sigma008_clip012    5759.4       95,000                 faster gait + larger residuals
ant_mpc_term001_ep1                     6054.5       100,000                terminal velocity cost
ant_mpc_default_adaptive_ep1            6146.2       106,300                speed-adaptive phase + stance
```

![Ant sample efficiency](heuristic_ant_sample_efficiency.png)

<video controls src="heuristic_ant_mpc_default_6146_render480.mp4" width="480"></video>

<details>
<summary>Reproduce the default Ant policy</summary>

This command reproduces the final default MPC policy. It is much slower than Breakout, because every real environment step evaluates `96 x 10` candidate residual actions inside a local MuJoCo model. The command below assumes the current directory is this article's artifact directory; `ant_envpool.xml` is already in the same directory as the script.

```bash
rm -f /tmp/repro_ant_6146_eval5.jsonl /tmp/repro_ant_6146_eval5.csv
python heuristic_ant.py \
  --policy mpc \
  --episodes 5 \
  --seed 0 \
  --max-steps 1000 \
  --mujoco-xml-path ant_envpool.xml \
  --trial-name repro_ant_6146_eval5 \
  --log-path /tmp/repro_ant_6146_eval5.jsonl \
  --summary-path /tmp/repro_ant_6146_eval5.csv
```

This command runs 5 episodes at once. It is much slower than the single-episode reproduction, but it shows some variance. The expected output should include `mean=6005.521`, `min=5776.805`, and `max=6146.208`. When I reran this command locally, the output was:

```text
episode=0 score=6146.208 x_position=285.434
episode=1 score=5982.507 x_position=277.088
episode=2 score=6028.890 x_position=279.226
episode=3 score=5776.805 x_position=267.084
episode=4 score=6093.194 x_position=282.733
eval_summary: episodes=5 env_steps=5000 mean=6005.521 min=5776.805 max=6146.208 x_mean=278.313 x_max=285.434
```

</details>

The full experiment script is in [`heuristic_ant.py`](heuristic_ant.py), the extracted minimal callable policy is in [`heuristic_ant_min_policy.py`](heuristic_ant_min_policy.py), and the experiment ledger is in [`heuristic_ant_trials_summary.csv`](heuristic_ant_trials_summary.csv).

The Ant example is different from Breakout. In Breakout, after the geometry was found, regularized input brought extreme sample efficiency. In Ant, a maintainable heuristic system kept getting more complex without turning into an uncontrollable pile of scripts. By the end, the policy had oscillator phase, stance ratio, speed adaptation, roll/pitch/yaw feedback, foot contacts, short-horizon model rollouts, residual smoothing, terminal velocity cost, and warm-start plan decay. A human could certainly write one or two of those modules. Maintaining the full experiment ledger, code, videos, and failed directions in a single day is a very different level of difficulty.

## Atari57: Throwing the Same Process at 57 Games

Breakout and Ant are two single-task stories. The Atari57 experiment was much more blunt: throw the same Codex workflow at the full Atari57 suite. For each environment, run both `ram` and `native_obs`, with 3 independent runs for each input type. In other words, each input type had `57 games x 3 runs = 171` coding-agent search trajectories.

I did not run it like the two examples above, where I sat in the Codex App, watched intermediate results, gave hints, corrected failures, and asked it to continue tuning. The setup was simple: use the Codex CLI to batch-launch a bunch of `gpt-5.4`, `xhigh` agents. Each agent got the same prompt template with different `ENV_ID / OBS_MODE / REPEAT_INDEX`, and then I did not touch it until the agents finished and stopped. To reduce hidden context, I did not use the Codex App, and I did not let it read long-term memory. It was a one-shot CLI batch run. The original Atari57 batch prompt is folded below, so the boundary is easy to inspect.

<details>
<summary>Original Atari57 batch prompt</summary>

````text
You are a pragmatic and rigorous coding agent. Focus only on EnvPool's {{ENV_ID}} under the {{OBS_MODE}} setting. Design and continuously iterate on a handwritten heuristic policy yourself, push the score as high as possible, and record sample efficiency completely. Run the whole process locally on the current machine. Do not search the internet. Do not reference any existing solution.

Task configuration:

ENV_ID = "{{ENV_ID}}"
OBS_MODE = "{{OBS_MODE}}"
REPEAT_INDEX = {{REPEAT_INDEX}}
KNOWN_BEST_SCORE = {{KNOWN_BEST_SCORE}}
ENV_ROOT_DIR = "/tmp/envpool_heuristic/{{ENV_ID}}"
ROOT_DIR = "/tmp/envpool_heuristic/{{ENV_ID}}/{{OBS_MODE}}"
RUN_DIR = "/tmp/envpool_heuristic/{{ENV_ID}}/{{OBS_MODE}}/run_{YYYYMMDD_HHMMSS}_{PID}"

ENVPOOL_VERSION = "1.1.1"
FRAME_BUDGET = 20000000

First create ENV_ROOT_DIR, ROOT_DIR, and RUN_DIR. Write every script, log, plot, and note produced by this run under RUN_DIR. Do not write into any repo workspace.

Before starting the experiment, do not create a venv and do not reinstall envpool. Directly run a minimal import check in the current Python environment and confirm that `envpool.__version__ == "1.1.1"`. Write the Python version, envpool version, envpool package path, and minimal import check result into RUN_DIR/README.md. If the current environment is not `envpool==1.1.1`, report the error and stop. Do not switch versions yourself.

Use REPEAT_INDEX as the initial random-seed offset, trial-name suffix, or another decorrelation mechanism for this search, so the 3 repeats do not mechanically become the exact same search trajectory.

Different tasks land under different ROOT_DIR / RUN_DIR paths, so the directories are naturally isolated. There is no need to serialize tasks just to avoid path conflicts. If useful, you may run non-overlapping exploration, evaluation, or analysis steps in parallel within this task.

Reference-score rule:

This prompt directly provides `KNOWN_BEST_SCORE`. Use it only for reference and result comparison, not for deciding when to stop.

Hard constraints:

1. Treat the environment strictly as a black box. Do not read the envpool source implementation for this environment/wrapper, test code, ROM/XML details, or any file that would leak internal environment implementation. Only use information exposed through EnvPool's public API: make/reset/step/render, action_space, observation_space, obs, reward, done, and public fields in info.

2. If OBS_MODE = "native_obs":
   - The policy may decide only from the native obs returned by reset/step and internal state maintained by the policy itself.
   - Do not read info["ram"] or any RAM content.
   - Do not use render() as an extra input to the policy.

3. If OBS_MODE = "ram":
   - The policy may read info["ram"] and internal state maintained by the policy itself.
   - Still do not call any unpublished simulator state or source-code internals.

4. If ENV_ID is an Atari environment, you must use the environment initialization template below. Do not change resolution, frame stack, frame skip, reward clipping, or sticky actions on your own:

```python
import envpool

env = envpool.make_gym(
    ENV_ID,
    num_envs=1,
    batch_size=1,
    seed=seed,
    img_height=210,
    img_width=160,
    stack_num=1,
    gray_scale=False,
    frame_skip=1,
    noop_max=1,
    use_fire_reset=True,
    episodic_life=False,
    reward_clip=False,
    repeat_action_probability=0.0,
    full_action_space=False,
)
```

5. If ENV_ID is not Atari, use that environment's default native obs initialization directly. Do not add wrappers unless the environment API explicitly requires them.

6. I am not constraining how you iterate, search, or organize the heuristic. You may use target detection, trajectory prediction, state machines, parameter search, short-horizon planning, controllers, or any other purely handwritten method you choose. The only requirements are: do not train a neural network, do not read environment source code, and do not read hidden state.

7. Do not stop to discuss a plan. Do not ask for confirmation. Do not output intermediate progress updates.

8. Whenever you refresh the current run's `best_score`, do not stop immediately. First enter a "code simplification phase":
   - The goal is to compress the heuristic into a shorter, more direct, easier-to-reproduce implementation without lowering the current `best_score`.
   - Prioritize deleting redundant search scripts, repeated branches, useless state, over-parameterization, and helper logic that does not contribute to the final score.
   - After simplification, you must evaluate again and confirm that `best_score` did not drop. If it drops, revert to the previous version that did not drop.
   - The final `policy.py` should be the simplest version corresponding to the current best score, not the bulkiest version that grew during search.

Stopping rule:

- For Atari tasks, the frame budget here is `FRAME_BUDGET = 20000000`. Because `frame_skip=1` is fixed, you can treat `cumulative_env_steps` directly as cumulative frames.
- While `cumulative_env_steps < FRAME_BUDGET`, do not stop because of score level, a short-term plateau, temporarily failing to find a better policy, or already exceeding/not exceeding KNOWN_BEST_SCORE. You must keep trying new heuristics, structures, searches, or evaluations.
- You may stop and output the final summary only when `cumulative_env_steps >= FRAME_BUDGET`.
- If the current environment is not Atari, still use the same `FRAME_BUDGET = 20000000` and interpret it as a cumulative env-step limit.

Output file requirements:

1. policy.py
   Save the current best heuristic. It should already have gone through the "simplification phase" and be as short and direct as possible without lowering best_score. Keep the interface simple, for example:

```python
class Policy:
    def reset(self):
        ...
    def act(self, obs: np.ndarray, info: dict | None = None):
        ...
```

2. trials.jsonl
   Append one line for every trial, containing at least:
   trial_index, timestamp, env_id, obs_mode, trial_name, episodes_finished, env_steps, score_mean, score_min, score_max, cumulative_env_steps, cumulative_episodes, policy_config, notes

3. summary.csv
   A summary generated from trials.jsonl.

4. sample_efficiency.png
   Draw two subplots from summary.csv:
   - x = cumulative_env_steps, y = score_mean and running_best
   - x = cumulative_episodes, y = score_mean and running_best

5. README.md
   At the end, write clearly: current best score, KNOWN_BEST_SCORE, REPEAT_INDEX, corresponding trial, cumulative env steps / episodes, FRAME_BUDGET, reproduction command, final simplified policy logic, main failed directions, and why you believe the stopping rule has been satisfied.

Sample accounting:

- Every trial / probe / debug rollout that actually steps the environment must count toward cumulative_env_steps and cumulative_episodes. Do not silently omit any of them.
- Score is episode return. If a trial runs multiple episodes, record mean / min / max.

Start now. First create RUN_DIR, probe ENV_ID's action_space, observation_space, reset obs shape, and step return structure, then decide how to iterate. Do not ask me questions or report progress midway. Output the final summary only after satisfying the stopping rule.
````

</details>

The Atari57 scores in the figure below came from this unattended batch run. I did not watch videos halfway through and pick directions. I did not feed a game's failure reason back into Codex and ask it to continue editing. Human intervention can obviously push single-task scores higher; the Breakout and Ant experiments already show that the upper bound can keep moving. But I do not think human intervention is a necessary condition for this route. It is more like an accelerator while current models are still not strong enough. As models get smarter, many steps like "watch the failure video, infer the missing mechanism, and keep editing code" should move directly into the unattended loop. Each run is like a small coding-agent researcher. It first probes actions, observations, and rewards, then generates candidate heuristic policies or local search scripts, then keeps evaluating, recording the best score, watching failure modes, and editing.

![Atari57 sample efficiency compared with OpenRL Benchmark](atari57_openrl_sample_efficiency_context.png)

The x-axis in this Atari57 figure starts at `10^4` environment steps because the earlier region is basically flat. The y-axis is Atari human-normalized score, or HNS. The Codex curves use the common Atari median convention: first take the median of the 3 independent runs for each game, then take the median across the 57 games. This statistic is not blown up by a few very high-scoring games, so it measures coverage: how many games this process has already found the key mechanism for.

In this fully unattended batch run, `native_obs` reached `0.81` around `9.7M` steps, and `ram` reached `0.59`. In the same figure, the PPO2 / CleanRL EnvPool PPO median HNS curves saved by [OpenRL Benchmark](https://arxiv.org/abs/2402.03046) are roughly `0.88` / `0.92` at `10M` steps. So the result is not "Codex has already beaten traditional reinforcement learning across the board." The more accurate statement is: a fairly rough coding-agent batch process, without looking at intermediate results at all, can already push the Atari57 median into the neighborhood of those baselines.

An aggregate curve compresses all differences into one median, so I also plotted the 57 games individually. Raw Atari returns are not comparable across games, so this still uses each game's own HNS; the dashed line at `1.0` is human score. The left panel sorts by heuristic score, and the right panel puts the heuristic and OpenRL's CleanRL EnvPool PPO at the same coordinate for each game.

![Atari57 per-game HNS comparison](atari57_per_game_hns_comparison.png)

This figure shows two things. First, there is overlap: in games like Breakout, Krull, DoubleDunk, Boxing, and DemonAttack, both the heuristic and the reinforcement-learning baseline get clearly above human score. Second, there are large differences too: the heuristic is relatively stronger on games such as Asterix, Jamesbond, Centipede, Bowling, Skiing, and Tennis; PPO is much stronger on Atlantis, VideoPinball, UpNDown, Assault, RoadRunner, and StarGunner. This distribution is more informative than a single median. The heuristic did not uniformly learn "how to play Atari"; in some games it quickly wrote down an effective mechanism, while in others it still had not found the right state representation or long-term strategy.

The accounting also needs to be clear. The Codex curves come from the raw `summary.csv` files of these 342 batch runs. I reorganized the full result set and reconstructed every search trajectory by `cumulative_env_steps`. This is still not a strict leaderboard: the compute Codex spent writing code, reading logs, and watching videos is not counted as neural-network training compute; the PPO curves are the median HNS curves saved by that benchmark. The comparison is mainly about one signal: under regularized inputs, heuristic policies maintained by coding agents can use very little environment interaction to push many games into a competitive range, and RAM and native observation are not wildly far apart.

The most interesting part of Atari57, to me, is that the source of sample efficiency changed. Traditional neural-network Atari learning has to learn representation, credit assignment, and action meanings again from high-dimensional input in every environment. Here, Codex decomposes the environment into a maintainable small-program system: aiming and dodging for shooters, bounces for paddle games, position rules for avoidance games, environment-wrapper details, and each environment's own failed-experiment ledger. It did not train a general neural network. It generated and maintained a batch of local heuristic/reflex systems.

## Why I Think This Is Worth Thinking About

Compared with neural networks, the two most obvious advantages of heuristic policies are sample efficiency and interpretability. The sample efficiency is already visible above: many times, Codex needed only tens of thousands to hundreds of thousands of steps to build a working policy structure. Interpretability matters just as much. Every Breakout improvement maps to a concrete mechanism: stuck ball paths, low fast balls, late-stage offset decay, paddle action delay. Every Ant improvement also maps to a concrete module: phase, yaw feedback, harmonics, residual planning, warm start. When something fails, you can open the code and the video and see which condition, which state estimate, or which action delay broke. That feels very different from staring at a reward curve.

Neural networks obviously have their own advantages, especially in expressivity, complex vision, and cross-state generalization. But this experiment made me think that many local control tasks do not need to be handed entirely to end-to-end training. A maintainable heuristic layer can first eat the "reflex actions" and "local geometry," leaving harder perception, long-horizon planning, and policy selection to learning systems. The system may not be simpler overall, but debugging becomes more like software engineering: logs, replays, ablations, and failure provenance.

The core issue is still maintenance cost. In the past, heuristic systems were hard to maintain because the maintainer had to be a programmer, experimenter, debugger, data recorder, and half a control engineer at the same time. A Breakout policy has to remember loop breaking, late-stage offset decay, paddle delay, visual detectors, and RAM mapping. An Ant policy has to remember joint order, phase, balance feedback, rollout objectives, planning budget, and failed configurations. It is hard for humans to maintain that much complexity over time, so it was natural for people to put their hopes into end-to-end training.

Coding agents make the tradeoff different. A heuristic system can become an "executable research note": state extraction, policy structure, parameter sweeps, failure diagnosis, video evidence, and sample-efficiency curves all live in the same code system. The policy is no longer just a fragile handwritten script. It can become a software artifact that an agent keeps maintaining, locally refactoring, and migrating across input formats.

## Implications for Robotics

There is still room to imagine robotics applications here, but the boundary needs to be clear. Ant is simulated locomotion, not a real robot. In simulation, Codex can try tens of thousands or millions of steps, and falling over just means reset. In the real world, every failed attempt costs time, hardware wear, safety margin, scene reset, and sensor drift. So this loop cannot simply be moved onto a real robot and run blindly the way it runs in simulation.

I would rather think of it as one layer of a robotics system, not a complete robotics solution. The parts that fit best are those where state can be observed reliably, failure can be rolled back safely, and the local control objective is clear: keep the body stable, return a joint to a safe angle, absorb force after foot contact, make an emergency recovery motion when falling, or limit force and speed as a gripper approaches an object. These are more like reflexes, and they may indeed be maintainable and regression-testable as programs.

Manipulation tasks are much harder. Folding clothes, organizing cables, or opening soft packaging is not just about how robot joints move. The object state itself matters: cloth deformation, occlusion, contact history, friction, wrinkles, target shape. Writing a heuristic policy for this is not the same as tuning a few joint phases. Without good perceptual representations, recoverable action primitives, and sufficiently realistic simulation, a coding agent will also write brittle rules on top of the wrong state variables.

So the more realistic form is probably a hybrid system: simulation and offline data first generate and filter candidate heuristics; the real robot only runs small, safe, guarded validations; neural networks handle perception, object-state estimation, and long-horizon value; heuristic/reflex systems handle low-latency safety constraints, local recovery, and test criteria; coding agents maintain interfaces, detectors, failure handling, and regression tests. In that form, the claim is not "handwrite cloth folding with heuristics." It is: pull some local regularities that can be written as programs out of end-to-end training.

## Limitations

Sample efficiency: the environment steps in the Atari57 figure are real policy-local EnvPool steps, but they do not include Codex's time spent writing code, reading logs, or watching videos. They are also not equivalent to a standard neural-network training budget. In Ant, residual model predictive planning also uses local rollouts inside a MuJoCo model, and that compute should not be mixed with real EnvPool environment steps.

RAM and image/native observations also need to be discussed separately. In the Atari57 aggregate result, the overall gap between RAM and native observations is not huge, and that is the most interesting phenomenon. But in single-game Breakout, the pure-image `14.5K` number is a transfer budget that inherits the geometric policy already found during the RAM stage. It shows that structure can be transferred and refactored by an agent. It should not be written as "pure image learned the maximum score from scratch in 14.5K steps."

Some environments are also a poor fit for reactive heuristic policies. Montezuma's Revenge is the clearest counterexample. In this experiment record, state-graph search could reduce the key distance from `72` to `28`, but the reward was still `0`. I later tried for a long time and eventually touched around `400` points; by then the policy really had learned a chain like "climb the ladder, jump over the enemy, pick up the key, open the door." The problem is that this kind of policy is awkward to express with ordinary code branches: actions need precise timing, failures need recovery, and intermediate states need to re-enter the plan. The limitation here feels more like the representation of the policy itself is not strong enough. Montezuma made me think the policy format matters too: some environments need composable macro-actions, recoverable search state, and maybe a program structure better suited to long-horizon planning than ordinary if-else code.

So the strongest conclusion here should be "promising," not "already replacing reinforcement-learning baselines." But the promise is real: once inputs are regularized, and once coding agents can maintain complex heuristic policies, sample efficiency can enter a regime that past neural-network learning curves rarely touched. That is what made me want to write this up.

## Conclusion

This experiment changed how I think about heuristic policies. I used to put them in the fragile, temporary, unscalable bucket. Now I am more inclined to see them as software systems that coding agents can keep maintaining. They no longer look like a few fixed rules. They look more like a set of programmable reflexes whose tests, visualizations, parameters, refactors, migrations, and provenance can all be maintained.

Atari57 shows that under regularized inputs, heuristic policies maintained by coding agents can use very few environment steps to push median HNS close to neural-network baselines, and that this can happen in an unattended batch run. It also exposes the current problem: search trajectories are heavy-tailed. Some games reveal their structure quickly, while others still do not. Breakout shows that with human intervention, the single-task upper bound can move higher: first discover geometric control, then maintain stuck handling and late-stage offset decay, and finally migrate the RAM state reader into an RGB detector. Ant shows that even in continuous control, Codex can start from a simple rhythmic gait, introduce residual model predictive planning on its own, and maintain a high-scoring multi-joint reflex system.

If I continue this line of work, I am no longer that interested in "raise one more game's score." I am more interested in making the process systematic: give a coding agent an environment, a strict experiment ledger, a sampling budget, and a rendering/diagnostic interface, then let it discover, maintain, and regression-test a heuristic/reflex layer on its own. This used to be too tiring for humans to maintain, so almost nobody took it seriously. With coding agents, it suddenly becomes a reasonable research direction. I also think that as models get smarter, the iteration time and cost will fall quickly, so generating a heuristic policy will become much cheaper than it is today.

The robotics version I most want to see is also narrower: start from safe, reproducible local reflexes and sim-to-real validation tools, instead of claiming that complex manipulation tasks can be handwritten directly. Real-world iteration is slow, and manipulation tasks involve object state, so this route is more likely to become part of training and testing systems: low-level reflexes handle safety and local feedback, while higher-level policies and perception models handle goals, object understanding, and composition. In that world, each joint does not necessarily need to be trained from zero, and each failure does not necessarily need to be absorbed by a larger network.

## Acknowledgements

Thanks to [Costa Huang](https://costa.sh/) and [Tairan He](https://tairanhe.com/) for feedback on this article.

## Citation

If you need to cite this article in LaTeX, you can start with the BibTeX below. After publication, replace `url` with the final link.

```bibtex
@misc{weng2026codex_heuristic_policy,
  title = {Make Heuristics Great Again: Letting Codex Build Heuristic Systems from Scratch},
  author = {Weng, Jiayi},
  year = {2026},
  month = apr,
  howpublished = {\url{https://example.com/codex-heuristic-policy}},
  note = {Blog post}
}
```

In the text, you can write:

```latex
\usepackage{url}

... as discussed in \cite{weng2026codex_heuristic_policy}.
```
