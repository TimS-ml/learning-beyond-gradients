#!/usr/bin/env python3
# ruff: noqa: D101,D103
"""Pure NumPy heuristic policy/search for Gymnasium HalfCheetah-v5.

The policies are intentionally not neural nets.  The current default is a PD
controller that tracks a Fourier target-angle gait.  A lower scoring fallback is
a Fourier torque generator plus four named proprioceptive reflex terms:

  action = gait(t) + joint_angle + joint_velocity + torso_pitch + pitch_rate

Use --search to keep tuning the selected policy parameters with CEM.  Every
CEM iteration writes a JSONL record containing sampled frames / episodes and
scores.

The optional mpc-asym-pd-cpg policy is still heuristic/no-NN: it evaluates
small one-action perturbations in copied MuJoCo states and falls back to the
asymmetric PD CPG for the lookahead tail.  It is diagnostic and intentionally
much slower than the default.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import gymnasium as gym
import mujoco
import mujoco.rollout
import numpy as np

ENV_ID = "HalfCheetah-v5"
EPISODE_STEPS = 1000
DT = 0.05

PolicyName = Literal[
    "open-loop",
    "reflex",
    "pd-cpg",
    "asym-pd-cpg",
    "mpc-asym-pd-cpg",
    "mpc-tree-asym-pd-cpg",
    "mpc-staged-tree-asym-pd-cpg",
]


# A robust open-loop gait found by CEM over:
# [freq_raw, bias[6], sin[6], cos[6], sin2[6], cos2[6]].
OPEN_LOOP_RAW = np.array(
    [
        2.2979370180260714,
        2.099256164873018,
        -0.5499759347609208,
        2.130070499224449,
        -0.9062253580837633,
        -1.5583878021055166,
        -1.4990378025632933,
        -1.0159493181974708,
        -0.8152562948965629,
        0.34002837961714805,
        -4.412866224819963,
        -1.3823178244862848,
        -1.7889499881830804,
        0.9208869712824361,
        -2.6943023434414526,
        1.782465150978837,
        -0.5786926527163252,
        -1.4195350563476838,
        1.5820487716702307,
        0.8607611237773094,
        0.713398492229306,
        0.26717997915694414,
        1.1033839101006184,
        1.2183397279977224,
        3.3892968036345654,
        -2.5010723030214956,
        1.9283954514882093,
        -0.35363904321129036,
        -1.641365020799348,
        -2.817348631010227,
        -0.7210679194745322,
    ],
    dtype=np.float64,
)


# Higher scoring reflex gait selected on 3 reset seeds per candidate.
# Raw layout = OPEN_LOOP_RAW-style first 31 scalars + 4 x 6 reflex raw gains.
REFLEX_RAW = np.array(
    [
        2.3482676038202603,
        2.352517498483182,
        -0.675398062021892,
        2.4145416855385275,
        -0.9610459240665964,
        -1.2208370250287892,
        -1.68741418711739,
        -1.053944427847365,
        -0.8648964642828982,
        0.43526018860996657,
        -4.7471369804676335,
        -1.2635778877338792,
        -1.8543304707749948,
        1.1649855371987101,
        -2.748833151062348,
        2.3671855500069965,
        -0.5861277970449001,
        -1.3583578218719061,
        1.4392341912669981,
        0.370763433705203,
        0.2837617244339373,
        0.023453931826339147,
        1.0316569812143823,
        1.2886791089655458,
        3.7403889678781614,
        -2.3954235535804074,
        1.8589781693999157,
        -0.3316010135207059,
        -1.6131355411019759,
        -3.0667142367278393,
        -0.7459741638677237,
        -0.47547394673673826,
        -0.33480738388500764,
        -0.1272339317799393,
        -0.13800269330734388,
        -0.5493670232100797,
        0.12932588427253958,
        0.18295573357974562,
        0.6393424017558722,
        0.17630644507282459,
        -0.1321861045276194,
        0.03388399823921129,
        0.10995255228705625,
        -0.7937385149904888,
        -0.10659083811441575,
        -0.22427880217742047,
        0.35009770429495857,
        0.08375589212893236,
        0.029629131323541835,
        -0.1519315254669959,
        0.17403826601671057,
        -0.23285493657803866,
        -0.2572752107717383,
        -0.033039914720600305,
        -1.1297565711591677,
    ],
    dtype=np.float64,
)


# PD controller selected by a small CEM over target-angle Fourier coefficients.
# Raw layout = [freq_raw, target_angle_coeff[5, 6].ravel()].
PD_CPG_RAW = np.array(
    [
        2.1487903116283396,
        0.8003315342646616,
        -0.12492708838250559,
        0.7356824837599555,
        -0.6939682425752202,
        -0.5982214912250471,
        -0.747897986862209,
        0.42755981198194837,
        -1.4088761311172673,
        0.7653653821318237,
        -0.6306469998942844,
        -0.7947520946583158,
        -0.41846522528854746,
        0.9634873751637475,
        -0.44155167963679615,
        0.5913523401379751,
        0.31536157700398426,
        -0.36872430829682984,
        -0.0699974745782837,
        0.24102532066004487,
        0.2558950763609027,
        -0.23892373390401722,
        0.307661719044352,
        0.34821179718471457,
        0.29630182888248247,
        -0.5014012308189947,
        -0.16524830884138347,
        -0.18569861368131627,
        -0.3228633670284166,
        -0.2632556015867013,
        -0.2603840965719395,
    ],
    dtype=np.float64,
)

# Same target-angle controller as `pd-cpg`, but the oscillator advances at a
# different frequency in the two half-cycles.  Raw layout =
# [stance_freq_raw, swing_freq_raw, target_angle_coeff[5, 6].ravel()].
ASYM_PD_CPG_RAW = np.array(
    [
        2.0720429776207037,
        2.0890962589603244,
        0.7454172634828156,
        -0.06577692937619559,
        0.6454089230776165,
        -0.6746024815060896,
        -0.5255276137795621,
        -0.776911900431121,
        0.4784639889085752,
        -1.5344226994990133,
        0.8642039688512475,
        -0.6508749004237321,
        -0.8139980805865409,
        -0.43273458554686073,
        0.9311256504045836,
        -0.3884941211535119,
        0.5058403771772146,
        0.29329785944557646,
        -0.41630444646878295,
        -0.17793318904003722,
        0.25287195521722133,
        0.1864602367775759,
        -0.2174643453198326,
        0.32053306369231493,
        0.18857020059058224,
        0.2877837436012836,
        -0.46494131953713835,
        -0.23362679675599532,
        -0.179250145837136,
        -0.2538333727428432,
        -0.19506084586961667,
        -0.3065628153040692,
    ],
    dtype=np.float64,
)

MPC_ASYM_PD_CPG_RAW = ASYM_PD_CPG_RAW.copy()
MPC_ASYM_PD_CPG_RAW[:2] -= 0.35
MPC_CRUISE_ASYM_PD_CPG_RAW = ASYM_PD_CPG_RAW.copy()
MPC_CRUISE_ASYM_PD_CPG_RAW[0] -= 1.80
MPC_CRUISE_ASYM_PD_CPG_RAW[1] -= 1.10


def scaled_mpc_cruise_raw(amplitude_scale: float, front_lower_leg_bias: float) -> np.ndarray:
    raw = MPC_CRUISE_ASYM_PD_CPG_RAW.copy()
    coeff = raw[2:].reshape(5, 6).copy()
    coeff[1:, :] *= amplitude_scale
    coeff[0, 4] += front_lower_leg_bias
    coeff[0, 5] += front_lower_leg_bias
    raw[2:] = coeff.reshape(-1)
    return raw


MPC_START_SWING_RAW = scaled_mpc_cruise_raw(amplitude_scale=1.15, front_lower_leg_bias=0.15)
MPC_FAST_SWING_RAW = scaled_mpc_cruise_raw(amplitude_scale=1.18, front_lower_leg_bias=0.20)
MPC_FINAL_SWING_RAW = MPC_START_SWING_RAW.copy()

PD_KP = 1.0
PD_KD = 0.02
MPC_PD_KP = 8.0
MPC_PD_KD = 0.0
ASYM_PHASE_MICROSTEPS = 10
MPC_HORIZON = 14
MPC_TERMINAL_VELOCITY_WEIGHT = 0.5
MPC_COORDINATE_DELTAS = (-0.10, 0.10, -0.25, 0.25, -0.50, 0.50, -0.75, 0.75, -1.00, 1.00)
MPC_RANDOM_BLOCKS = ((0.15, 64), (0.35, 128), (0.70, 192))
MPC_BANG_BANG_ACTIONS = np.asarray(np.meshgrid(*[[-1.0, 1.0]] * 6), dtype=np.float64).T.reshape(-1, 6)
MPC_TREE_TOP_K = 8
MPC_FAST_SWING_SWITCH_STEP = 300
MPC_FINAL_SWING_SWITCH_STEP = 900
MODEL_FRAME_SKIP = 5


@dataclass(frozen=True)
class Evaluation:
    episodes: int
    frames: int
    mean_return: float
    std_return: float
    min_return: float
    max_return: float
    returns: list[float]


@dataclass(frozen=True)
class SearchIteration:
    iteration: int
    sampled_frames_this_iter: int
    sampled_episodes_this_iter: int
    sampled_frames_total: int
    sampled_episodes_total: int
    batch_best_score: float
    batch_best_mean_return: float
    batch_best_returns: list[float]
    batch_mean_score: float
    best_score: float
    best_mean_return: float
    best_returns: list[float]
    search_action: str


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-x))


def decode_gait(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Decode raw gait params for one or many candidates."""
    raw = np.asarray(raw, dtype=np.float64)
    is_one = raw.ndim == 1
    raw2 = raw[None, :] if is_one else raw
    freq = 0.5 + 4.5 * sigmoid(raw2[:, 0])
    coeff = 0.9 * np.tanh(raw2[:, 1:31].reshape(-1, 5, 6))
    if is_one:
        return freq[0], coeff[0]
    return freq, coeff


def fourier_action(raw31: np.ndarray, t: int) -> np.ndarray:
    freq, coeff = decode_gait(raw31)
    theta = 2.0 * math.pi * float(freq) * t * DT
    features = np.array(
        [1.0, math.sin(theta), math.cos(theta), math.sin(2 * theta), math.cos(2 * theta)],
        dtype=np.float64,
    )
    return features @ coeff


def reflex_action(raw55: np.ndarray, obs: np.ndarray, t: int) -> np.ndarray:
    action = fourier_action(raw55[:31], t)
    feedback = raw55[31:].reshape(4, 6)
    joint_angle_gain = 0.5 * np.tanh(feedback[0])
    joint_velocity_gain = 0.12 * np.tanh(feedback[1])
    pitch_gain = 0.45 * np.tanh(feedback[2])
    pitch_rate_gain = 0.12 * np.tanh(feedback[3])

    joint_angle = obs[2:8]
    joint_velocity = obs[11:17]
    pitch = obs[1]
    pitch_rate = obs[10]
    return (
        action
        + joint_angle_gain * joint_angle
        + joint_velocity_gain * joint_velocity
        + pitch_gain * pitch
        + pitch_rate_gain * pitch_rate
    )


def pd_cpg_action(raw31: np.ndarray, obs: np.ndarray, t: int) -> np.ndarray:
    freq, _ = decode_gait(np.concatenate([raw31[:1], np.zeros(30, dtype=np.float64)]))
    theta = 2.0 * math.pi * float(freq) * t * DT
    features = np.array(
        [1.0, math.sin(theta), math.cos(theta), math.sin(2 * theta), math.cos(2 * theta)],
        dtype=np.float64,
    )
    target_joint_angle = features @ raw31[1:].reshape(5, 6)
    return PD_KP * (target_joint_angle - obs[2:8]) - PD_KD * obs[11:17]


def pd_cpg_action_batch(raw31: np.ndarray, obs: np.ndarray, steps: np.ndarray) -> np.ndarray:
    raw31 = np.asarray(raw31, dtype=np.float64)
    freq, _ = decode_gait(
        np.concatenate([raw31[:, :1], np.zeros((len(raw31), 30), dtype=np.float64)], axis=1)
    )
    theta = 2.0 * math.pi * freq * steps * DT
    features = np.stack(
        [np.ones_like(theta), np.sin(theta), np.cos(theta), np.sin(2 * theta), np.cos(2 * theta)],
        axis=1,
    )
    target_joint_angle = np.einsum("nf,nfa->na", features, raw31[:, 1:].reshape(-1, 5, 6))
    action = PD_KP * (target_joint_angle - obs[:, 2:8]) - PD_KD * obs[:, 11:17]
    return np.clip(action, -1.0, 1.0).astype(np.float32)


def advance_asym_phase(raw32: np.ndarray, phase: np.ndarray) -> np.ndarray:
    raw32 = np.asarray(raw32, dtype=np.float64)
    stance_freq, _ = decode_gait(
        np.concatenate([raw32[:, 0:1], np.zeros((len(raw32), 30), dtype=np.float64)], axis=1)
    )
    swing_freq, _ = decode_gait(
        np.concatenate([raw32[:, 1:2], np.zeros((len(raw32), 30), dtype=np.float64)], axis=1)
    )
    for _ in range(ASYM_PHASE_MICROSTEPS):
        freq = np.where(np.sin(phase) > 0.0, swing_freq, stance_freq)
        phase = (phase + 2.0 * math.pi * freq * (DT / ASYM_PHASE_MICROSTEPS)) % (2.0 * math.pi)
    return phase


def asym_pd_cpg_action(
    raw32: np.ndarray,
    obs: np.ndarray,
    phase: float,
    kp: float = PD_KP,
    kd: float = PD_KD,
) -> tuple[np.ndarray, float]:
    phase_arr = advance_asym_phase(raw32[None, :], np.asarray([phase], dtype=np.float64))
    features = np.array(
        [
            1.0,
            math.sin(float(phase_arr[0])),
            math.cos(float(phase_arr[0])),
            math.sin(2.0 * float(phase_arr[0])),
            math.cos(2.0 * float(phase_arr[0])),
        ],
        dtype=np.float64,
    )
    target_joint_angle = features @ raw32[2:].reshape(5, 6)
    action = kp * (target_joint_angle - obs[2:8]) - kd * obs[11:17]
    return action, float(phase_arr[0])


def asym_pd_cpg_action_batch(
    raw32: np.ndarray,
    obs: np.ndarray,
    phase: np.ndarray,
    kp: float = PD_KP,
    kd: float = PD_KD,
) -> tuple[np.ndarray, np.ndarray]:
    phase = advance_asym_phase(raw32, phase)
    features = np.stack(
        [np.ones_like(phase), np.sin(phase), np.cos(phase), np.sin(2 * phase), np.cos(2 * phase)],
        axis=1,
    )
    target_joint_angle = np.einsum("nf,nfa->na", features, raw32[:, 2:].reshape(-1, 5, 6))
    action = kp * (target_joint_angle - obs[:, 2:8]) - kd * obs[:, 11:17]
    return np.clip(action, -1.0, 1.0).astype(np.float32), phase


def model_obs(data: mujoco.MjData) -> np.ndarray:
    return np.concatenate([data.qpos[1:].copy(), data.qvel.copy()])


def model_obs_batch(model: mujoco.MjModel, state: np.ndarray) -> np.ndarray:
    qpos_start = 1
    qvel_start = 1 + model.nq
    qpos = state[:, qpos_start:qvel_start]
    qvel = state[:, qvel_start : qvel_start + model.nv]
    return np.concatenate([qpos[:, 1:], qvel], axis=1)


def model_state(data: mujoco.MjData) -> np.ndarray:
    return np.concatenate([[data.time], data.qpos.copy(), data.qvel.copy()]).astype(np.float64)


def copy_model_data(model: mujoco.MjModel, source: mujoco.MjData, target: mujoco.MjData) -> None:
    target.time = source.time
    target.qpos[:] = source.qpos
    target.qvel[:] = source.qvel
    target.ctrl[:] = source.ctrl
    mujoco.mj_forward(model, target)


def step_model_env(model: mujoco.MjModel, data: mujoco.MjData, action: np.ndarray) -> float:
    x_before = float(data.qpos[0])
    data.ctrl[:] = action
    for _ in range(MODEL_FRAME_SKIP):
        mujoco.mj_step(model, data)
    velocity_reward = (float(data.qpos[0]) - x_before) / DT
    control_cost = 0.1 * float(np.sum(action * action))
    return velocity_reward - control_cost


def step_model_env_batch(
    model: mujoco.MjModel,
    rollout_data: mujoco.MjData,
    state: np.ndarray,
    action: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Step many copied model states by one Gym env step and return reward."""
    control = np.repeat(action[:, None, :], MODEL_FRAME_SKIP, axis=1).astype(np.float64)
    states, _ = mujoco.rollout.rollout(
        model,
        rollout_data,
        initial_state=state,
        control=control,
    )
    next_state = states[:, -1, :]
    velocity_reward = (next_state[:, 1] - state[:, 1]) / DT
    control_cost = 0.1 * np.sum(action * action, axis=1)
    return next_state, velocity_reward - control_cost


def score_first_action_with_baseline_tail(
    raw32: np.ndarray,
    env_data: mujoco.MjData,
    phase: float,
    first_action: np.ndarray,
    scratch_data: mujoco.MjData,
    model: mujoco.MjModel,
    kp: float = MPC_PD_KP,
    kd: float = MPC_PD_KD,
) -> float:
    copy_model_data(model, env_data, scratch_data)
    simulated_phase = float(advance_asym_phase(raw32[None, :], np.asarray([phase], dtype=np.float64))[0])
    score = step_model_env(model, scratch_data, first_action)
    for _ in range(1, MPC_HORIZON):
        action, simulated_phase = asym_pd_cpg_action(raw32, model_obs(scratch_data), simulated_phase, kp, kd)
        score += step_model_env(model, scratch_data, np.clip(action, -1.0, 1.0))
    return score + MPC_TERMINAL_VELOCITY_WEIGHT * float(scratch_data.qvel[0])


def score_first_actions_with_baseline_tail_batch(
    raw32: np.ndarray,
    env_data: mujoco.MjData,
    phase: float,
    first_actions: np.ndarray,
    rollout_data: mujoco.MjData,
    model: mujoco.MjModel,
    kp: float = MPC_PD_KP,
    kd: float = MPC_PD_KD,
) -> np.ndarray:
    """Vectorized equivalent of score_first_action_with_baseline_tail."""
    action_count = len(first_actions)
    raw_repeated = np.repeat(raw32[None, :], action_count, axis=0)
    state = np.repeat(model_state(env_data)[None, :], action_count, axis=0)
    simulated_phase = advance_asym_phase(
        raw_repeated,
        np.full(action_count, phase, dtype=np.float64),
    )

    action = np.clip(first_actions, -1.0, 1.0).astype(np.float64)
    state, score = step_model_env_batch(model, rollout_data, state, action)
    for _ in range(1, MPC_HORIZON):
        action, simulated_phase = asym_pd_cpg_action_batch(
            raw_repeated,
            model_obs_batch(model, state),
            simulated_phase,
            kp,
            kd,
        )
        state, reward = step_model_env_batch(model, rollout_data, state, action.astype(np.float64))
        score += reward
    qvel_start = 1 + model.nq
    return score + MPC_TERMINAL_VELOCITY_WEIGHT * state[:, qvel_start]


def score_actions_from_states_with_baseline_tail_batch(
    raw32: np.ndarray,
    model: mujoco.MjModel,
    rollout_data: mujoco.MjData,
    initial_state: np.ndarray,
    phase: np.ndarray,
    first_actions: np.ndarray,
    kp: float = MPC_PD_KP,
    kd: float = MPC_PD_KD,
) -> np.ndarray:
    """Score first actions from already-copied model states."""
    action_count = len(first_actions)
    raw_repeated = np.repeat(raw32[None, :], action_count, axis=0)
    simulated_phase = advance_asym_phase(raw_repeated, np.asarray(phase, dtype=np.float64).copy())

    action = np.clip(first_actions, -1.0, 1.0).astype(np.float64)
    state, score = step_model_env_batch(model, rollout_data, initial_state, action)
    for _ in range(1, MPC_HORIZON):
        action, simulated_phase = asym_pd_cpg_action_batch(
            raw_repeated,
            model_obs_batch(model, state),
            simulated_phase,
            kp,
            kd,
        )
        state, reward = step_model_env_batch(model, rollout_data, state, action.astype(np.float64))
        score += reward
    qvel_start = 1 + model.nq
    return score + MPC_TERMINAL_VELOCITY_WEIGHT * state[:, qvel_start]


def mpc_candidate_actions(
    base_action: np.ndarray,
    rng: np.random.Generator,
    *,
    include_random: bool,
) -> np.ndarray:
    candidate_actions = [base_action]
    for joint_index in range(6):
        for delta in MPC_COORDINATE_DELTAS:
            action = base_action.copy()
            action[joint_index] = np.clip(action[joint_index] + delta, -1.0, 1.0)
            candidate_actions.append(action)
    if include_random:
        for std, count in MPC_RANDOM_BLOCKS:
            random_actions = base_action + std * rng.standard_normal((count, 6))
            candidate_actions.extend(np.clip(random_actions, -1.0, 1.0))
    candidate_actions.extend(MPC_BANG_BANG_ACTIONS)
    return np.asarray(candidate_actions, dtype=np.float64)


def mpc_asym_pd_cpg_action(
    raw32: np.ndarray,
    obs: np.ndarray,
    phase: float,
    env: gym.Env,
    rollout_data: mujoco.MjData,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float]:
    base_action, next_phase = asym_pd_cpg_action(raw32, obs, phase, MPC_PD_KP, MPC_PD_KD)
    base_action = np.clip(base_action, -1.0, 1.0)
    candidates = mpc_candidate_actions(base_action, rng, include_random=True)

    unwrapped = env.unwrapped
    scores = score_first_actions_with_baseline_tail_batch(
        raw32,
        unwrapped.data,
        phase,
        candidates,
        rollout_data,
        unwrapped.model,
    )
    return candidates[int(np.argmax(scores))], next_phase


def mpc_tree_asym_pd_cpg_action(
    raw32: np.ndarray,
    obs: np.ndarray,
    phase: float,
    env: gym.Env,
    rollout_data: mujoco.MjData,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float]:
    base_action, next_phase = asym_pd_cpg_action(raw32, obs, phase, MPC_PD_KP, MPC_PD_KD)
    base_action = np.clip(base_action, -1.0, 1.0)
    candidates = mpc_candidate_actions(base_action, rng, include_random=True)

    unwrapped = env.unwrapped
    model = unwrapped.model
    scores = score_first_actions_with_baseline_tail_batch(
        raw32,
        unwrapped.data,
        phase,
        candidates,
        rollout_data,
        model,
    )
    top_candidate_indices = np.argsort(scores)[-MPC_TREE_TOP_K:]
    top_first_actions = candidates[top_candidate_indices]

    top_k = len(top_first_actions)
    first_state = np.repeat(model_state(unwrapped.data)[None, :], top_k, axis=0)
    second_state, first_reward = step_model_env_batch(
        model,
        rollout_data,
        first_state,
        top_first_actions.astype(np.float64),
    )
    second_phase = np.full(top_k, next_phase, dtype=np.float64)
    raw_repeated = np.repeat(raw32[None, :], top_k, axis=0)
    second_base_actions, _ = asym_pd_cpg_action_batch(
        raw_repeated,
        model_obs_batch(model, second_state),
        second_phase.copy(),
        MPC_PD_KP,
        MPC_PD_KD,
    )

    second_actions_by_branch = [
        mpc_candidate_actions(second_base_actions[branch].astype(np.float64), rng, include_random=False)
        for branch in range(top_k)
    ]
    branch_indices = np.concatenate(
        [np.full(len(actions), branch, dtype=np.int64) for branch, actions in enumerate(second_actions_by_branch)]
    )
    second_actions = np.concatenate(second_actions_by_branch, axis=0)
    second_initial_state = np.concatenate(
        [np.repeat(second_state[branch : branch + 1], len(actions), axis=0) for branch, actions in enumerate(second_actions_by_branch)],
        axis=0,
    )
    second_initial_phase = np.concatenate(
        [np.full(len(actions), second_phase[branch], dtype=np.float64) for branch, actions in enumerate(second_actions_by_branch)]
    )
    tree_scores = first_reward[branch_indices] + score_actions_from_states_with_baseline_tail_batch(
        raw32,
        model,
        rollout_data,
        second_initial_state,
        second_initial_phase,
        second_actions,
    )
    return top_first_actions[branch_indices[int(np.argmax(tree_scores))]], next_phase


def action_batch(policy: PolicyName, raw: np.ndarray, obs: np.ndarray, steps: np.ndarray) -> np.ndarray:
    """Vectorized policy for search/eval."""
    if policy == "pd-cpg":
        return pd_cpg_action_batch(raw, obs, steps)

    raw55 = raw
    freq, coeff = decode_gait(raw55[:, :31])
    theta = 2.0 * math.pi * freq * steps * DT
    features = np.stack(
        [np.ones_like(theta), np.sin(theta), np.cos(theta), np.sin(2 * theta), np.cos(2 * theta)],
        axis=1,
    )
    action = np.einsum("nf,nfa->na", features, coeff)

    feedback = raw55[:, 31:].reshape(-1, 4, 6)
    action += 0.5 * np.tanh(feedback[:, 0]) * obs[:, 2:8]
    action += 0.12 * np.tanh(feedback[:, 1]) * obs[:, 11:17]
    action += 0.45 * np.tanh(feedback[:, 2]) * obs[:, 1:2]
    action += 0.12 * np.tanh(feedback[:, 3]) * obs[:, 10:11]
    return np.clip(action, -1.0, 1.0).astype(np.float32)


def default_raw(policy: PolicyName) -> np.ndarray:
    if policy == "open-loop":
        raw = np.zeros(55, dtype=np.float64)
        raw[:31] = OPEN_LOOP_RAW
        return raw
    if policy == "reflex":
        return REFLEX_RAW.copy()
    if policy == "pd-cpg":
        return PD_CPG_RAW.copy()
    if policy == "asym-pd-cpg":
        return ASYM_PD_CPG_RAW.copy()
    if policy in {
        "mpc-asym-pd-cpg",
        "mpc-tree-asym-pd-cpg",
        "mpc-staged-tree-asym-pd-cpg",
    }:
        return MPC_ASYM_PD_CPG_RAW.copy()
    raise ValueError(f"unknown policy: {policy}")


def rollout(policy: PolicyName, seed: int, render: bool = False) -> float:
    render_mode = "human" if render else None
    env = gym.make(ENV_ID, render_mode=render_mode)
    obs, _ = env.reset(seed=seed)
    total = 0.0
    raw = default_raw(policy)
    asym_phase = 0.0
    mpc_rollout_data = mujoco.MjData(env.unwrapped.model)
    mpc_rng = np.random.default_rng(seed + 333)
    for t in range(EPISODE_STEPS):
        if policy == "open-loop":
            action = fourier_action(raw[:31], t)
        elif policy == "mpc-staged-tree-asym-pd-cpg":
            if t < MPC_FAST_SWING_SWITCH_STEP:
                staged_raw = MPC_START_SWING_RAW
            elif t < MPC_FINAL_SWING_SWITCH_STEP:
                staged_raw = MPC_FAST_SWING_RAW
            else:
                staged_raw = MPC_FINAL_SWING_RAW
            action, asym_phase = mpc_tree_asym_pd_cpg_action(
                staged_raw,
                obs,
                asym_phase,
                env,
                mpc_rollout_data,
                mpc_rng,
            )
        elif policy == "mpc-tree-asym-pd-cpg":
            action, asym_phase = mpc_tree_asym_pd_cpg_action(
                raw,
                obs,
                asym_phase,
                env,
                mpc_rollout_data,
                mpc_rng,
            )
        elif policy == "mpc-asym-pd-cpg":
            action, asym_phase = mpc_asym_pd_cpg_action(
                raw,
                obs,
                asym_phase,
                env,
                mpc_rollout_data,
                mpc_rng,
            )
        elif policy == "asym-pd-cpg":
            action, asym_phase = asym_pd_cpg_action(raw, obs, asym_phase)
        elif policy == "pd-cpg":
            action = pd_cpg_action(raw, obs, t)
        else:
            action = reflex_action(raw, obs, t)
        obs, reward, terminated, truncated, _ = env.step(np.clip(action, -1.0, 1.0).astype(np.float32))
        total += float(reward)
        if terminated or truncated:
            break
    env.close()
    return total


def evaluate_policy(policy: PolicyName, episodes: int, seed: int) -> Evaluation:
    returns = [rollout(policy, seed + episode) for episode in range(episodes)]
    returns_arr = np.asarray(returns, dtype=np.float64)
    return Evaluation(
        episodes=episodes,
        frames=episodes * EPISODE_STEPS,
        mean_return=float(np.mean(returns_arr)),
        std_return=float(np.std(returns_arr)),
        min_return=float(np.min(returns_arr)),
        max_return=float(np.max(returns_arr)),
        returns=[float(x) for x in returns],
    )


def evaluate_raw_candidates(
    policy: PolicyName,
    raw: np.ndarray,
    repeats: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return stable score, mean return, returns per candidate/repeat."""
    candidates = np.asarray(raw, dtype=np.float64)
    candidate_count = len(candidates)
    tiled = np.repeat(candidates, repeats, axis=0)
    env_count = len(tiled)

    env = gym.make_vec(ENV_ID, num_envs=env_count, vectorization_mode="sync")
    obs, _ = env.reset(seed=seed)
    returns = np.zeros(env_count, dtype=np.float64)
    steps = np.zeros(env_count, dtype=np.float64)
    asym_phase = np.zeros(env_count, dtype=np.float64)
    for t in range(EPISODE_STEPS):
        if policy == "asym-pd-cpg":
            action, asym_phase = asym_pd_cpg_action_batch(tiled, obs, asym_phase)
        else:
            action = action_batch(policy, tiled, obs, steps)
        obs, reward, terminated, truncated, _ = env.step(action)
        returns += reward
        steps += 1.0
        if t < EPISODE_STEPS - 1 and np.any(terminated | truncated):
            raise RuntimeError(f"{ENV_ID} ended before {EPISODE_STEPS} steps in vector eval")
    env.close()

    returns = returns.reshape(candidate_count, repeats)
    mean_return = np.mean(returns, axis=1)
    score = mean_return - 0.20 * np.std(returns, axis=1)
    return score, mean_return, returns


def append_jsonl(path: Path, record: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def search(args: argparse.Namespace) -> np.ndarray:
    if args.policy in {
        "mpc-asym-pd-cpg",
        "mpc-tree-asym-pd-cpg",
        "mpc-staged-tree-asym-pd-cpg",
    }:
        raise ValueError(f"{args.policy} is an online model-based overlay; search asym-pd-cpg instead")

    rng = np.random.default_rng(args.seed)
    mean = default_raw(args.policy)
    std = np.full_like(mean, args.gait_std, dtype=np.float64)
    std[31:] = args.reflex_std

    best_raw = mean.copy()
    best_score, best_mean, best_returns = evaluate_raw_candidates(
        args.policy,
        best_raw[None, :],
        args.repeats,
        args.eval_seed,
    )
    best_score_float = float(best_score[0])

    sampled_frames_total = args.repeats * EPISODE_STEPS
    sampled_episodes_total = args.repeats
    log_path = Path(args.search_log)

    for iteration in range(1, args.search_iterations + 1):
        start = time.time()
        raw = mean + std * rng.standard_normal((args.population, len(mean)))
        raw[0] = best_raw

        score, mean_return, returns = evaluate_raw_candidates(
            args.policy,
            raw,
            args.repeats,
            seed=args.eval_seed + 997 * iteration,
        )

        sampled_frames_this_iter = args.population * args.repeats * EPISODE_STEPS
        sampled_episodes_this_iter = args.population * args.repeats
        sampled_frames_total += sampled_frames_this_iter
        sampled_episodes_total += sampled_episodes_this_iter

        order = np.argsort(score)[::-1]
        best_batch_index = int(order[0])
        if float(score[best_batch_index]) > best_score_float:
            best_raw = raw[best_batch_index].copy()
            best_score_float = float(score[best_batch_index])
            best_mean = mean_return[best_batch_index : best_batch_index + 1]
            best_returns = returns[best_batch_index : best_batch_index + 1]

        elites = raw[order[: args.elites]]
        new_mean = np.mean(elites, axis=0)
        new_std = np.std(elites, axis=0) + args.std_bonus
        mean = args.cem_alpha * new_mean + (1.0 - args.cem_alpha) * mean
        std = np.maximum(
            args.cem_alpha * new_std + (1.0 - args.cem_alpha) * std,
            args.min_std,
        )

        record = SearchIteration(
            iteration=iteration,
            sampled_frames_this_iter=sampled_frames_this_iter,
            sampled_episodes_this_iter=sampled_episodes_this_iter,
            sampled_frames_total=sampled_frames_total,
            sampled_episodes_total=sampled_episodes_total,
            batch_best_score=float(score[best_batch_index]),
            batch_best_mean_return=float(mean_return[best_batch_index]),
            batch_best_returns=[float(x) for x in returns[best_batch_index]],
            batch_mean_score=float(np.mean(score)),
            best_score=best_score_float,
            best_mean_return=float(best_mean[0]),
            best_returns=[float(x) for x in best_returns[0]],
            search_action=(
                "CEM: keep top "
                f"{args.elites}/{args.population}; {args.repeats} reset seeds/candidate; "
                f"elapsed_s={time.time() - start:.2f}"
            ),
        )
        append_jsonl(log_path, asdict(record))
        print(json.dumps(asdict(record), sort_keys=True))

    return best_raw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--policy",
        choices=[
            "open-loop",
            "reflex",
            "pd-cpg",
            "asym-pd-cpg",
            "mpc-asym-pd-cpg",
            "mpc-tree-asym-pd-cpg",
            "mpc-staged-tree-asym-pd-cpg",
        ],
        default="asym-pd-cpg",
    )
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--eval-seed", type=int, default=100)
    parser.add_argument("--render", action="store_true")

    parser.add_argument("--search", action="store_true")
    parser.add_argument("--search-iterations", type=int, default=10)
    parser.add_argument("--population", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--elites", type=int, default=10)
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--search-log", default="examples/heuristic_halfcheetah_v5_search.jsonl")
    parser.add_argument("--save-best", default="")
    parser.add_argument("--cem-alpha", type=float, default=0.5)
    parser.add_argument("--gait-std", type=float, default=0.18)
    parser.add_argument("--reflex-std", type=float, default=0.35)
    parser.add_argument("--std-bonus", type=float, default=0.02)
    parser.add_argument("--min-std", type=float, default=0.035)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.render:
        score = rollout(args.policy, args.eval_seed, render=True)
        print(json.dumps({"policy": args.policy, "seed": args.eval_seed, "return": score}))
        return

    if args.search:
        best_raw = search(args)
        if args.save_best:
            Path(args.save_best).parent.mkdir(parents=True, exist_ok=True)
            np.save(args.save_best, best_raw)
        return

    evaluation = evaluate_policy(args.policy, args.eval_episodes, args.eval_seed)
    print(json.dumps({"policy": args.policy, **asdict(evaluation)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
