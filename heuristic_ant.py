# Copyright 2021 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Play Ant-v5 with hand-written rhythmic and MPC-residual gait controllers.

The `rhythmic` policy is a low-dimensional oscillator over desired joint
angles, tracked by a PD controller using only the observation vector. The
`mpc` policy keeps the same oscillator as a stabilizing prior and uses a
short-horizon MuJoCo rollout to search a tiny residual action sequence online.
Neither mode uses a trained model or reward/info fields from EnvPool to choose
actions.

Every script invocation appends one JSONL record to
`heuristic_ant_trials.jsonl` and rewrites
`heuristic_ant_trials_summary.csv` with cumulative sample counts.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_LOG_PATH = SCRIPT_DIR / "heuristic_ant_trials.jsonl"
DEFAULT_SUMMARY_PATH = SCRIPT_DIR / "heuristic_ant_trials_summary.csv"
DEFAULT_MUJOCO_XML_PATH = (
    SCRIPT_DIR.parent
    / "third_party"
    / "mujoco_gym_xml_patches"
    / "ant_envpool.xml"
)

ANT_Q_INDEX = np.asarray([6, 7, 0, 1, 2, 3, 4, 5], dtype=np.int64)
ANT_LEG_PHASE = np.asarray([0.0, math.pi, 0.0, math.pi], dtype=np.float64)
ANT_HIP_SIGN = np.asarray([-1.0, 1.0, 1.0, -1.0], dtype=np.float64)
ANT_ANKLE_SIGN = np.asarray([1.0, 1.0, -1.0, -1.0], dtype=np.float64)
ANT_HEADING_AXIS = np.asarray([1.0, 1.0, -1.0, -1.0], dtype=np.float64)
ANT_PITCH_AXIS = np.asarray([1.0, 1.0, -1.0, -1.0], dtype=np.float64)
ANT_ROLL_AXIS = np.asarray([-1.0, 1.0, 1.0, -1.0], dtype=np.float64)
ANT_FOOT_BODY_IDS = np.asarray([13, 4, 7, 10], dtype=np.int64)
ANT_FOOT_OBS_ROWS = ANT_FOOT_BODY_IDS - 1


@dataclass(frozen=True)
class AntPolicyConfig:
    """Parameters for one rhythmic PD gait."""

    dphi: float = 0.660934259732249
    dphi_speed_gain: float = -0.02
    dphi_speed_target: float = 5.8
    dphi_min: float = 0.62
    dphi_max: float = 0.72
    hip_bias: float = 0.12217781430672398
    hip_amp: float = 0.5705418365199333
    ankle_bias: float = 0.36651046903486795
    ankle_amp: float = 0.26587749767314783
    kp: float = 0.8108143632989734
    kd: float = 0.0
    pitch_gain: float = -0.19444590796726124
    pitch_rate_gain: float = 0.04099276700871415
    roll_gain: float = -0.25536960225655303
    roll_rate_gain: float = 0.023293075237761272
    contact_hip_gain: float = 0.0
    contact_ankle_gain: float = 0.0
    contact_push_hip_amp: float = 0.0
    contact_push_ankle_amp: float = 0.0
    contact_push_phase: float = 2.356194490192345
    contact_push_width: float = 0.55
    stance_duty: float = 0.6355364206196007
    stance_duty_speed_gain: float = -0.01
    stance_duty_speed_target: float = 5.8
    stance_duty_min: float = 0.6
    stance_duty_max: float = 0.67
    hip_stance_scale: float = 1.0479076970107701
    hip_swing_scale: float = 1.0031777685985328
    ankle_stance_scale: float = 0.976603459922793
    ankle_swing_scale: float = 0.9374473230114526
    yaw_gain: float = -0.12067720879887742
    yaw_rate_gain: float = 0.04418873596679619
    hip_h2_amp: float = 0.10975404801587477
    hip_h2_phase: float = 2.0862256065597453
    ankle_h2_amp: float = -0.003434817287963554
    ankle_h2_phase: float = 1.2927488104774438
    hip_h3_amp: float = 0.04827596673280693
    hip_h3_phase: float = -0.49944083263433436
    ankle_h3_amp: float = -0.06968988354403895
    ankle_h3_phase: float = 1.5873441034476188
    mpc_horizon: int = 10
    mpc_candidates: int = 96
    mpc_cem_iters: int = 0
    mpc_elite_frac: float = 0.125
    mpc_num_knots: int = 0
    mpc_sigma: float = 0.07614211639071694
    mpc_clip: float = 0.12016284361036686
    mpc_pose_cost: float = 23.348190567885954
    mpc_pitch_target: float = 0.0
    mpc_yaw_cost: float = 2.7292168081366723
    mpc_z_cost: float = 2.1215830559511737
    mpc_z_target: float = 0.4519975076600261
    mpc_forward_weight: float = 1.0
    mpc_ctrl_cost: float = 0.5
    mpc_terminal_vel_cost: float = 0.01
    mpc_plan_decay: float = 0.504186948858276
    mpc_seed: int = 12
    mujoco_xml_path: str = str(DEFAULT_MUJOCO_XML_PATH)


@dataclass
class AntPolicyState:
    """Minimal recurrent state for one rollout."""

    phase: float = 0.0


def warp_ant_leg_phase(leg_phase: np.ndarray, stance_duty: float) -> np.ndarray:
    """Stretch the stance half-cycle and compress the swing half-cycle."""
    clipped_duty = float(np.clip(stance_duty, 0.05, 0.95))
    if abs(clipped_duty - 0.5) < 1e-12:
        return np.mod(leg_phase, 2.0 * math.pi)

    phase_unit = np.mod(leg_phase, 2.0 * math.pi) / (2.0 * math.pi)
    stance_unit = 0.5 * phase_unit / clipped_duty
    swing_unit = (
        0.5
        + 0.5 * (phase_unit - clipped_duty) / (1.0 - clipped_duty)
    )
    warped_unit = np.where(
        phase_unit < clipped_duty,
        stance_unit,
        swing_unit,
    )
    return 2.0 * math.pi * warped_unit


def compute_adaptive_ant_dphi(
    config: AntPolicyConfig,
    x_velocity: float,
) -> float:
    """Return one phase increment, optionally adapted to forward speed."""
    dphi = config.dphi + config.dphi_speed_gain * (
        x_velocity - config.dphi_speed_target
    )
    return float(np.clip(dphi, config.dphi_min, config.dphi_max))


def compute_adaptive_ant_stance_duty(
    config: AntPolicyConfig,
    x_velocity: float,
) -> float:
    """Return one stance duty, optionally adapted to forward speed."""
    stance_duty = config.stance_duty + config.stance_duty_speed_gain * (
        x_velocity - config.stance_duty_speed_target
    )
    return float(
        np.clip(
            stance_duty,
            config.stance_duty_min,
            config.stance_duty_max,
        )
    )


def compute_rhythmic_ant_action(
    config: AntPolicyConfig,
    phase: float,
    stance_duty: float,
    q: np.ndarray,
    dq: np.ndarray,
    roll: float,
    pitch: float,
    yaw: float,
    roll_rate: float,
    pitch_rate: float,
    yaw_rate: float,
    foot_contacts: np.ndarray,
) -> np.ndarray:
    """Return one PD-tracked oscillator action in actuator order."""
    leg_phase = warp_ant_leg_phase(
        phase + ANT_LEG_PHASE,
        stance_duty,
    )
    stance_mask = leg_phase < math.pi
    hip_wave = (
        config.hip_bias
        + np.where(
            stance_mask,
            config.hip_stance_scale,
            config.hip_swing_scale,
        )
        * (
            config.hip_amp * np.sin(leg_phase)
            + config.hip_h2_amp
            * np.sin(2.0 * leg_phase + config.hip_h2_phase)
            + config.hip_h3_amp
            * np.sin(3.0 * leg_phase + config.hip_h3_phase)
        )
    )
    ankle_wave = (
        config.ankle_bias
        + np.where(
            stance_mask,
            config.ankle_stance_scale,
            config.ankle_swing_scale,
        )
        * (
            config.ankle_amp * np.cos(leg_phase)
            + config.ankle_h2_amp
            * np.cos(2.0 * leg_phase + config.ankle_h2_phase)
            + config.ankle_h3_amp
            * np.cos(3.0 * leg_phase + config.ankle_h3_phase)
        )
    )
    push_width = max(config.contact_push_width, 1e-3)
    contact_push = (
        np.exp(
            -0.5
            * np.square(
                (leg_phase - config.contact_push_phase) / push_width
            )
        )
        * stance_mask
        * foot_contacts
    )
    hip_wave = hip_wave + config.contact_push_hip_amp * contact_push
    ankle_wave = ankle_wave + config.contact_push_ankle_amp * contact_push
    balance_wave = (
        ANT_PITCH_AXIS
        * (config.pitch_gain * pitch + config.pitch_rate_gain * pitch_rate)
        - ANT_ROLL_AXIS
        * (config.roll_gain * roll + config.roll_rate_gain * roll_rate)
    )
    action = np.empty(8, dtype=np.float64)
    action[0::2] = config.kp * (
        ANT_HIP_SIGN * hip_wave
        + ANT_HEADING_AXIS
        * (
            config.yaw_gain * yaw
            + config.yaw_rate_gain * yaw_rate
            + config.contact_hip_gain * foot_contacts
        )
        - q[0::2]
    ) - config.kd * dq[0::2]
    action[1::2] = (
        config.kp
        * (
            ANT_ANKLE_SIGN
            * (
                ankle_wave
                + balance_wave
                + config.contact_ankle_gain * foot_contacts
            )
            - q[1::2]
        )
        - config.kd * dq[1::2]
    )
    return np.clip(action, -1.0, 1.0)


def sample_mpc_residual_noise(
    config: AntPolicyConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample one smooth residual-noise tensor in `(horizon, action_dim)`."""
    if config.mpc_num_knots <= 1:
        residuals = rng.normal(
            0.0,
            config.mpc_sigma,
            size=(config.mpc_horizon, 8),
        )
        residuals[1:] = 0.6 * residuals[1:] + 0.4 * residuals[:-1]
        return residuals

    num_knots = int(np.clip(config.mpc_num_knots, 2, config.mpc_horizon))
    knot_t = np.linspace(0.0, config.mpc_horizon - 1, num_knots)
    step_t = np.arange(config.mpc_horizon, dtype=np.float64)
    knot_noise = rng.normal(0.0, config.mpc_sigma, size=(num_knots, 8))
    residuals = np.empty((config.mpc_horizon, 8), dtype=np.float64)
    for action_idx in range(8):
        residuals[:, action_idx] = np.interp(
            step_t,
            knot_t,
            knot_noise[:, action_idx],
        )
    return residuals


class RhythmicAntAgent:
    """Left-right anti-phase joint-space oscillator tracked by PD control."""

    def __init__(self, config: AntPolicyConfig) -> None:
        self._config = config
        self._state = AntPolicyState()

    def reset(self) -> None:
        self._state = AntPolicyState()
        self._rng = np.random.default_rng(self._config.mpc_seed)

    def act(self, obs: np.ndarray) -> np.ndarray:
        q, dq = decode_ant_joint_state(obs)
        x_velocity = decode_ant_forward_velocity(obs)
        roll, pitch, yaw, roll_rate, pitch_rate, yaw_rate = (
            decode_ant_torso_state(obs)
        )
        foot_contacts = decode_ant_foot_contacts(obs)
        stance_duty = compute_adaptive_ant_stance_duty(
            self._config,
            x_velocity,
        )
        action = compute_rhythmic_ant_action(
            self._config,
            self._state.phase,
            stance_duty,
            q[ANT_Q_INDEX],
            dq[ANT_Q_INDEX],
            roll,
            pitch,
            yaw,
            roll_rate,
            pitch_rate,
            yaw_rate,
            foot_contacts,
        )
        self._state.phase += compute_adaptive_ant_dphi(
            self._config,
            x_velocity,
        )
        return action


class MpcResidualAntAgent:
    """Short-horizon residual shooting around the rhythmic controller."""

    def __init__(self, config: AntPolicyConfig) -> None:
        self._config = config
        self._state = AntPolicyState()
        self._rng = np.random.default_rng(config.mpc_seed)
        try:
            import mujoco
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "MPC mode requires `mujoco`; install it before "
                "running `--policy mpc`."
            ) from e

        self._mujoco = mujoco
        self._model = mujoco.MjModel.from_xml_path(config.mujoco_xml_path)
        self._rollout_data = mujoco.MjData(self._model)
        self._residual_plan = np.zeros(
            (config.mpc_horizon, 8),
            dtype=np.float64,
        )

    def reset(self) -> None:
        self._state = AntPolicyState()
        self._rng = np.random.default_rng(self._config.mpc_seed)
        self._residual_plan.fill(0.0)

    def act(self, obs: np.ndarray) -> np.ndarray:
        self._set_mujoco_state_from_obs(obs)
        q = self._rollout_data.qpos[7:15][ANT_Q_INDEX]
        dq = self._rollout_data.qvel[6:14][ANT_Q_INDEX]
        x_velocity = decode_ant_forward_velocity(obs)
        roll, pitch, yaw, roll_rate, pitch_rate, yaw_rate = (
            decode_ant_torso_state(obs)
        )
        foot_contacts = decode_ant_foot_contacts(obs)
        stance_duty = compute_adaptive_ant_stance_duty(
            self._config,
            x_velocity,
        )
        base_action = compute_rhythmic_ant_action(
            self._config,
            self._state.phase,
            stance_duty,
            q,
            dq,
            roll,
            pitch,
            yaw,
            roll_rate,
            pitch_rate,
            yaw_rate,
            foot_contacts,
        )
        best_residuals = self._residual_plan.copy()
        best_objective = self._rollout_objective(obs, best_residuals)
        if self._config.mpc_cem_iters <= 0:
            for _ in range(self._config.mpc_candidates - 1):
                if self._config.mpc_num_knots <= 1:
                    residuals = np.clip(
                        best_residuals
                        + self._rng.normal(
                            0.0,
                            self._config.mpc_sigma,
                            size=(self._config.mpc_horizon, 8),
                        ),
                        -self._config.mpc_clip,
                        self._config.mpc_clip,
                    )
                    residuals[1:] = (
                        0.6 * residuals[1:] + 0.4 * residuals[:-1]
                    )
                else:
                    residuals = np.clip(
                        best_residuals
                        + sample_mpc_residual_noise(
                            self._config,
                            self._rng,
                        ),
                        -self._config.mpc_clip,
                        self._config.mpc_clip,
                    )
                objective = self._rollout_objective(obs, residuals)
                if objective > best_objective:
                    best_objective = objective
                    best_residuals = residuals
        else:
            residual_mean = best_residuals.copy()
            residual_std = np.full(
                (self._config.mpc_horizon, 8),
                self._config.mpc_sigma,
                dtype=np.float64,
            )
            num_candidates = max(2, self._config.mpc_candidates)
            num_elites = int(
                np.clip(
                    round(num_candidates * self._config.mpc_elite_frac),
                    2,
                    num_candidates,
                )
            )
            for _ in range(self._config.mpc_cem_iters):
                candidates = np.clip(
                    residual_mean[None, :, :]
                    + self._rng.normal(
                        0.0,
                        residual_std[None, :, :],
                        size=(num_candidates, self._config.mpc_horizon, 8),
                    ),
                    -self._config.mpc_clip,
                    self._config.mpc_clip,
                )
                candidates[:, 1:, :] = (
                    0.6 * candidates[:, 1:, :]
                    + 0.4 * candidates[:, :-1, :]
                )
                candidates[0, :, :] = residual_mean

                objectives = np.empty(num_candidates, dtype=np.float64)
                for candidate_idx, residuals in enumerate(candidates):
                    objective = self._rollout_objective(obs, residuals)
                    objectives[candidate_idx] = objective
                    if objective > best_objective:
                        best_objective = objective
                        best_residuals = residuals.copy()

                elite_indices = np.argpartition(
                    objectives,
                    -num_elites,
                )[-num_elites:]
                elite_residuals = candidates[elite_indices]
                residual_mean = np.mean(elite_residuals, axis=0)
                residual_std = np.maximum(
                    0.35 * np.std(elite_residuals, axis=0),
                    0.02 * self._config.mpc_sigma,
                )

        self._state.phase += compute_adaptive_ant_dphi(
            self._config,
            x_velocity,
        )
        self._residual_plan[:-1] = (
            self._config.mpc_plan_decay * best_residuals[1:]
        )
        self._residual_plan[-1] = 0.0
        return np.clip(base_action + best_residuals[0], -1.0, 1.0)

    def _set_mujoco_state_from_obs(self, obs: np.ndarray) -> None:
        arr = np.asarray(obs)
        if arr.ndim == 2:
            arr = arr[0]
        if arr.ndim != 1 or arr.shape[0] < 27:
            raise ValueError(f"Unsupported Ant observation shape: {arr.shape}")
        self._rollout_data.qpos[0:2] = 0.0
        self._rollout_data.qpos[2:] = arr[:13]
        self._rollout_data.qvel[:] = arr[13:27]
        self._mujoco.mj_forward(self._model, self._rollout_data)

    def _rollout_objective(
        self,
        obs: np.ndarray,
        residuals: np.ndarray,
    ) -> float:
        self._set_mujoco_state_from_obs(obs)
        objective = 0.0
        phase = self._state.phase
        for horizon_idx, residual in enumerate(residuals):
            q = self._rollout_data.qpos[7:15][ANT_Q_INDEX]
            dq = self._rollout_data.qvel[6:14][ANT_Q_INDEX]
            x_velocity_before = float(self._rollout_data.qvel[0])
            roll, pitch, yaw = decode_mujoco_torso_euler(
                self._rollout_data.qpos[3:7]
            )
            roll_rate = float(self._rollout_data.qvel[3])
            pitch_rate = float(self._rollout_data.qvel[4])
            yaw_rate = float(self._rollout_data.qvel[5])
            if horizon_idx == 0:
                foot_contacts = decode_ant_foot_contacts(obs)
            else:
                foot_contacts = decode_mujoco_foot_contacts(
                    self._rollout_data.cfrc_ext
                )
            stance_duty = compute_adaptive_ant_stance_duty(
                self._config,
                x_velocity_before,
            )
            action = np.clip(
                compute_rhythmic_ant_action(
                    self._config,
                    phase,
                    stance_duty,
                    q,
                    dq,
                    roll,
                    pitch,
                    yaw,
                    roll_rate,
                    pitch_rate,
                    yaw_rate,
                    foot_contacts,
                )
                + residual,
                -1.0,
                1.0,
            )
            x_before = float(self._rollout_data.qpos[0])
            self._rollout_data.ctrl[:] = action
            for _ in range(5):
                self._mujoco.mj_step(self._model, self._rollout_data)

            x_velocity = (
                float(self._rollout_data.qpos[0]) - x_before
            ) / (5.0 * self._model.opt.timestep)
            z_position = float(self._rollout_data.qpos[2])
            roll, pitch, yaw = decode_mujoco_torso_euler(
                self._rollout_data.qpos[3:7]
            )
            objective += self._config.mpc_forward_weight * x_velocity + (
                1.0 if 0.2 <= z_position <= 1.0 else -50.0
            )
            objective -= self._config.mpc_ctrl_cost * float(
                np.square(action).sum()
            )
            objective -= self._config.mpc_pose_cost * (
                roll * roll
                + (pitch - self._config.mpc_pitch_target)
                * (pitch - self._config.mpc_pitch_target)
            )
            objective -= self._config.mpc_yaw_cost * yaw * yaw
            objective -= self._config.mpc_z_cost * (
                z_position - self._config.mpc_z_target
            ) ** 2
            if z_position < 0.23 or z_position > 0.95:
                objective -= 100.0
            phase += compute_adaptive_ant_dphi(
                self._config,
                x_velocity,
            )

        qvel = self._rollout_data.qvel[6:14]
        objective -= self._config.mpc_terminal_vel_cost * float(
            np.dot(qvel, qvel)
        )
        return objective


def decode_ant_joint_state(obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return one env's 8 hinge positions and velocities in qpos order."""
    arr = np.asarray(obs)
    if arr.ndim == 2:
        arr = arr[0]
    if arr.ndim != 1 or arr.shape[0] < 27:
        raise ValueError(f"Unsupported Ant observation shape: {arr.shape}")
    return arr[5:13], arr[19:27]


def decode_ant_forward_velocity(obs: np.ndarray) -> float:
    """Return torso x velocity from one env observation."""
    arr = np.asarray(obs)
    if arr.ndim == 2:
        arr = arr[0]
    if arr.ndim != 1 or arr.shape[0] < 14:
        raise ValueError(f"Unsupported Ant observation shape: {arr.shape}")
    return float(arr[13])


def decode_ant_torso_state(
    obs: np.ndarray,
) -> tuple[float, float, float, float, float, float]:
    """Return torso roll/pitch/yaw angles and angular rates."""
    arr = np.asarray(obs)
    if arr.ndim == 2:
        arr = arr[0]
    if arr.ndim != 1 or arr.shape[0] < 19:
        raise ValueError(f"Unsupported Ant observation shape: {arr.shape}")

    quat = np.asarray(arr[1:5], dtype=np.float64)
    quat /= np.linalg.norm(quat) + 1e-12
    roll, pitch, yaw = decode_mujoco_torso_euler(quat)
    return roll, pitch, yaw, float(arr[16]), float(arr[17]), float(arr[18])


def decode_ant_foot_contacts(obs: np.ndarray) -> np.ndarray:
    """Return clipped vertical contact force proxies in actuator-pair order."""
    arr = np.asarray(obs)
    if arr.ndim == 2:
        arr = arr[0]
    if arr.ndim != 1:
        raise ValueError(f"Unsupported Ant observation shape: {arr.shape}")
    if arr.shape[0] < 27 + 13 * 6:
        return np.zeros(4, dtype=np.float64)

    cfrc_ext = arr[27 : 27 + 13 * 6].reshape(13, 6)
    return np.clip(cfrc_ext[ANT_FOOT_OBS_ROWS, 5], 0.0, 1.0)


def decode_mujoco_foot_contacts(cfrc_ext: np.ndarray) -> np.ndarray:
    """Return clipped vertical contact force proxies from MuJoCo body forces."""
    return np.clip(cfrc_ext[ANT_FOOT_BODY_IDS, 5], 0.0, 1.0)


def decode_mujoco_torso_euler(quat: np.ndarray) -> tuple[float, float, float]:
    """Return roll, pitch, yaw from a MuJoCo torso quaternion."""
    quat = np.asarray(quat, dtype=np.float64)
    quat /= np.linalg.norm(quat) + 1e-12
    w, x, y, z = quat
    roll = math.atan2(
        2.0 * (w * x + y * z),
        1.0 - 2.0 * (x * x + y * y),
    )
    pitch = math.asin(
        np.clip(
            2.0 * (w * y - z * x),
            -1.0,
            1.0,
        )
    )
    yaw = math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )
    return roll, pitch, yaw


def reset_env_with_info(env):
    """Return `(obs, info)` for Gymnasium or `(obs, {})` for legacy Gym."""
    result = env.reset()
    if isinstance(result, tuple):
        return result
    return result, {}


def step_env(env, action: np.ndarray):
    """Support both legacy Gym and Gymnasium step signatures."""
    result = env.step(action[None, :])
    if len(result) == 5:
        obs, reward, terminated, truncated, info = result
        done = np.logical_or(terminated, truncated)
    else:
        obs, reward, done, info = result
    return obs, float(np.asarray(reward)[0]), bool(np.asarray(done)[0]), info


def append_trial_record(log_path: Path, record: dict[str, Any]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def load_trial_records(log_path: Path) -> list[dict[str, Any]]:
    if not log_path.exists():
        return []
    records = []
    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_summary(
    log_path: Path,
    summary_path: Path,
) -> list[dict[str, Any]]:
    records = load_trial_records(log_path)
    rows = []
    cumulative_env_steps = 0
    for index, record in enumerate(records):
        env_steps = int(record.get("env_steps", 0))
        cumulative_env_steps += env_steps
        rows.append(
            {
                "trial_index": index,
                "timestamp": record.get("timestamp", ""),
                "trial_name": record.get("trial_name", ""),
                "kind": record.get("kind", ""),
                "num_envs": record.get("num_envs", 1),
                "episodes_started": record.get("episodes_started", 0),
                "episodes_finished": record.get("episodes_finished", 0),
                "env_steps": env_steps,
                "cumulative_env_steps": cumulative_env_steps,
                "score_mean": record.get("score_mean", ""),
                "score_min": record.get("score_min", ""),
                "score_max": record.get("score_max", ""),
                "x_position_mean": record.get("x_position_mean", ""),
                "x_position_max": record.get("x_position_max", ""),
                "notes": record.get("notes", ""),
            }
        )

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "trial_index",
                "timestamp",
                "trial_name",
                "kind",
                "num_envs",
                "episodes_started",
                "episodes_finished",
                "env_steps",
                "cumulative_env_steps",
                "score_mean",
                "score_min",
                "score_max",
                "x_position_mean",
                "x_position_max",
                "notes",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return rows


def print_summary(rows: list[dict[str, Any]]) -> None:
    if not rows:
        print("summary: no trial records")
        return
    last_row = rows[-1]
    scored_rows = [
        row for row in rows if row["score_mean"] not in ("", None)
    ]
    best_row = None
    if scored_rows:
        best_row = max(scored_rows, key=lambda row: float(row["score_mean"]))

    print(
        "trial_log_summary:",
        f"records={len(rows)}",
        f"cumulative_env_steps={last_row['cumulative_env_steps']}",
    )
    if best_row is not None:
        print(
            "best_mean_score:",
            f"trial_index={best_row['trial_index']}",
            f"trial_name={best_row['trial_name']}",
            f"score_mean={float(best_row['score_mean']):.3f}",
            f"x_position_mean={float(best_row['x_position_mean']):.3f}",
            f"cumulative_env_steps={best_row['cumulative_env_steps']}",
        )


def evaluate_heuristic_policy(args: argparse.Namespace) -> None:
    """Run the Ant policy and append one trial record."""
    import envpool

    env = envpool.make_gym(
        "Ant-v5",
        num_envs=1,
        batch_size=1,
        seed=args.seed,
        max_episode_steps=args.max_steps,
    )
    config = AntPolicyConfig(
        dphi=args.dphi,
        dphi_speed_gain=args.dphi_speed_gain,
        dphi_speed_target=args.dphi_speed_target,
        dphi_min=args.dphi_min,
        dphi_max=args.dphi_max,
        hip_bias=args.hip_bias,
        hip_amp=args.hip_amp,
        ankle_bias=args.ankle_bias,
        ankle_amp=args.ankle_amp,
        kp=args.kp,
        kd=args.kd,
        pitch_gain=args.pitch_gain,
        pitch_rate_gain=args.pitch_rate_gain,
        roll_gain=args.roll_gain,
        roll_rate_gain=args.roll_rate_gain,
        contact_hip_gain=args.contact_hip_gain,
        contact_ankle_gain=args.contact_ankle_gain,
        contact_push_hip_amp=args.contact_push_hip_amp,
        contact_push_ankle_amp=args.contact_push_ankle_amp,
        contact_push_phase=args.contact_push_phase,
        contact_push_width=args.contact_push_width,
        stance_duty=args.stance_duty,
        stance_duty_speed_gain=args.stance_duty_speed_gain,
        stance_duty_speed_target=args.stance_duty_speed_target,
        stance_duty_min=args.stance_duty_min,
        stance_duty_max=args.stance_duty_max,
        hip_stance_scale=args.hip_stance_scale,
        hip_swing_scale=args.hip_swing_scale,
        ankle_stance_scale=args.ankle_stance_scale,
        ankle_swing_scale=args.ankle_swing_scale,
        yaw_gain=args.yaw_gain,
        yaw_rate_gain=args.yaw_rate_gain,
        hip_h2_amp=args.hip_h2_amp,
        hip_h2_phase=args.hip_h2_phase,
        ankle_h2_amp=args.ankle_h2_amp,
        ankle_h2_phase=args.ankle_h2_phase,
        hip_h3_amp=args.hip_h3_amp,
        hip_h3_phase=args.hip_h3_phase,
        ankle_h3_amp=args.ankle_h3_amp,
        ankle_h3_phase=args.ankle_h3_phase,
        mpc_horizon=args.mpc_horizon,
        mpc_candidates=args.mpc_candidates,
        mpc_cem_iters=args.mpc_cem_iters,
        mpc_elite_frac=args.mpc_elite_frac,
        mpc_num_knots=args.mpc_num_knots,
        mpc_sigma=args.mpc_sigma,
        mpc_clip=args.mpc_clip,
        mpc_pose_cost=args.mpc_pose_cost,
        mpc_pitch_target=args.mpc_pitch_target,
        mpc_yaw_cost=args.mpc_yaw_cost,
        mpc_z_cost=args.mpc_z_cost,
        mpc_z_target=args.mpc_z_target,
        mpc_forward_weight=args.mpc_forward_weight,
        mpc_ctrl_cost=args.mpc_ctrl_cost,
        mpc_terminal_vel_cost=args.mpc_terminal_vel_cost,
        mpc_plan_decay=args.mpc_plan_decay,
        mpc_seed=args.mpc_seed,
        mujoco_xml_path=args.mujoco_xml_path,
    )
    if args.policy == "rhythmic":
        agent = RhythmicAntAgent(config)
    elif args.policy == "mpc":
        agent = MpcResidualAntAgent(config)
    else:
        raise ValueError(f"Unsupported policy: {args.policy}")

    scores = []
    x_positions = []
    env_steps = 0
    episodes_started = 0
    episodes_finished = 0

    for episode in range(args.episodes):
        obs, info = reset_env_with_info(env)
        agent.reset()
        episode_score = 0.0
        x_position = 0.0
        episodes_started += 1
        for _ in range(args.max_steps):
            action = agent.act(obs)
            obs, reward, done, info = step_env(env, action)
            episode_score += reward
            if "x_position" in info:
                x_position = float(np.asarray(info["x_position"])[0])
            env_steps += 1
            if done:
                break

        scores.append(episode_score)
        x_positions.append(x_position)
        episodes_finished += 1
        print(
            f"episode={episode}",
            f"score={episode_score:.3f}",
            f"x_position={x_position:.3f}",
        )

    print(
        "eval_summary:",
        f"episodes={episodes_finished}",
        f"env_steps={env_steps}",
        f"mean={np.mean(scores):.3f}",
        f"min={np.min(scores):.3f}",
        f"max={np.max(scores):.3f}",
        f"x_mean={np.mean(x_positions):.3f}",
        f"x_max={np.max(x_positions):.3f}",
    )

    log_path = Path(args.log_path)
    summary_path = Path(args.summary_path)
    append_trial_record(
        log_path,
        {
            "timestamp": datetime.now().astimezone().isoformat(
                timespec="seconds"
            ),
            "trial_name": args.trial_name,
            "kind": "eval",
            "num_envs": 1,
            "episodes_started": episodes_started,
            "episodes_finished": episodes_finished,
            "env_steps": env_steps,
            "score_mean": float(np.mean(scores)),
            "score_min": float(np.min(scores)),
            "score_max": float(np.max(scores)),
            "x_position_mean": float(np.mean(x_positions)),
            "x_position_max": float(np.max(x_positions)),
            "scores": scores,
            "x_positions": x_positions,
            "params": {
                "policy": args.policy,
                "dphi": args.dphi,
                "dphi_speed_gain": args.dphi_speed_gain,
                "dphi_speed_target": args.dphi_speed_target,
                "dphi_min": args.dphi_min,
                "dphi_max": args.dphi_max,
                "hip_bias": args.hip_bias,
                "hip_amp": args.hip_amp,
                "ankle_bias": args.ankle_bias,
                "ankle_amp": args.ankle_amp,
                "kp": args.kp,
                "kd": args.kd,
                "pitch_gain": args.pitch_gain,
                "pitch_rate_gain": args.pitch_rate_gain,
                "roll_gain": args.roll_gain,
                "roll_rate_gain": args.roll_rate_gain,
                "contact_hip_gain": args.contact_hip_gain,
                "contact_ankle_gain": args.contact_ankle_gain,
                "contact_push_hip_amp": args.contact_push_hip_amp,
                "contact_push_ankle_amp": args.contact_push_ankle_amp,
                "contact_push_phase": args.contact_push_phase,
                "contact_push_width": args.contact_push_width,
                "stance_duty": args.stance_duty,
                "stance_duty_speed_gain": args.stance_duty_speed_gain,
                "stance_duty_speed_target": args.stance_duty_speed_target,
                "stance_duty_min": args.stance_duty_min,
                "stance_duty_max": args.stance_duty_max,
                "hip_stance_scale": args.hip_stance_scale,
                "hip_swing_scale": args.hip_swing_scale,
                "ankle_stance_scale": args.ankle_stance_scale,
                "ankle_swing_scale": args.ankle_swing_scale,
                "yaw_gain": args.yaw_gain,
                "yaw_rate_gain": args.yaw_rate_gain,
                "hip_h2_amp": args.hip_h2_amp,
                "hip_h2_phase": args.hip_h2_phase,
                "ankle_h2_amp": args.ankle_h2_amp,
                "ankle_h2_phase": args.ankle_h2_phase,
                "hip_h3_amp": args.hip_h3_amp,
                "hip_h3_phase": args.hip_h3_phase,
                "ankle_h3_amp": args.ankle_h3_amp,
                "ankle_h3_phase": args.ankle_h3_phase,
                "mpc_horizon": args.mpc_horizon,
                "mpc_candidates": args.mpc_candidates,
                "mpc_cem_iters": args.mpc_cem_iters,
                "mpc_elite_frac": args.mpc_elite_frac,
                "mpc_num_knots": args.mpc_num_knots,
                "mpc_sigma": args.mpc_sigma,
                "mpc_clip": args.mpc_clip,
                "mpc_pose_cost": args.mpc_pose_cost,
                "mpc_pitch_target": args.mpc_pitch_target,
                "mpc_yaw_cost": args.mpc_yaw_cost,
                "mpc_z_cost": args.mpc_z_cost,
                "mpc_z_target": args.mpc_z_target,
                "mpc_forward_weight": args.mpc_forward_weight,
                "mpc_ctrl_cost": args.mpc_ctrl_cost,
                "mpc_terminal_vel_cost": args.mpc_terminal_vel_cost,
                "mpc_plan_decay": args.mpc_plan_decay,
                "mpc_seed": args.mpc_seed,
                "mujoco_xml_path": args.mujoco_xml_path,
            },
            "notes": args.notes,
        },
    )
    rows = write_summary(log_path, summary_path)
    print_summary(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a pure heuristic Ant-v5 gait policy."
    )
    parser.add_argument(
        "--policy",
        type=str,
        default="rhythmic",
        choices=["rhythmic", "mpc"],
    )
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--dphi", type=float, default=0.660934259732249)
    parser.add_argument("--dphi-speed-gain", type=float, default=-0.02)
    parser.add_argument("--dphi-speed-target", type=float, default=5.8)
    parser.add_argument("--dphi-min", type=float, default=0.62)
    parser.add_argument("--dphi-max", type=float, default=0.72)
    parser.add_argument("--hip-bias", type=float, default=0.12217781430672398)
    parser.add_argument("--hip-amp", type=float, default=0.5705418365199333)
    parser.add_argument("--ankle-bias", type=float, default=0.36651046903486795)
    parser.add_argument("--ankle-amp", type=float, default=0.26587749767314783)
    parser.add_argument("--kp", type=float, default=0.8108143632989734)
    parser.add_argument("--kd", type=float, default=0.0)
    parser.add_argument(
        "--pitch-gain",
        type=float,
        default=-0.19444590796726124,
    )
    parser.add_argument(
        "--pitch-rate-gain",
        type=float,
        default=0.04099276700871415,
    )
    parser.add_argument(
        "--roll-gain",
        type=float,
        default=-0.25536960225655303,
    )
    parser.add_argument(
        "--roll-rate-gain",
        type=float,
        default=0.023293075237761272,
    )
    parser.add_argument("--contact-hip-gain", type=float, default=0.0)
    parser.add_argument("--contact-ankle-gain", type=float, default=0.0)
    parser.add_argument("--contact-push-hip-amp", type=float, default=0.0)
    parser.add_argument("--contact-push-ankle-amp", type=float, default=0.0)
    parser.add_argument(
        "--contact-push-phase",
        type=float,
        default=2.356194490192345,
    )
    parser.add_argument("--contact-push-width", type=float, default=0.55)
    parser.add_argument(
        "--stance-duty",
        type=float,
        default=0.6355364206196007,
    )
    parser.add_argument(
        "--stance-duty-speed-gain",
        type=float,
        default=-0.01,
    )
    parser.add_argument(
        "--stance-duty-speed-target",
        type=float,
        default=5.8,
    )
    parser.add_argument(
        "--stance-duty-min",
        type=float,
        default=0.6,
    )
    parser.add_argument(
        "--stance-duty-max",
        type=float,
        default=0.67,
    )
    parser.add_argument(
        "--hip-stance-scale",
        type=float,
        default=1.0479076970107701,
    )
    parser.add_argument(
        "--hip-swing-scale",
        type=float,
        default=1.0031777685985328,
    )
    parser.add_argument(
        "--ankle-stance-scale",
        type=float,
        default=0.976603459922793,
    )
    parser.add_argument(
        "--ankle-swing-scale",
        type=float,
        default=0.9374473230114526,
    )
    parser.add_argument("--yaw-gain", type=float, default=-0.12067720879887742)
    parser.add_argument("--yaw-rate-gain", type=float, default=0.04418873596679619)
    parser.add_argument("--hip-h2-amp", type=float, default=0.10975404801587477)
    parser.add_argument("--hip-h2-phase", type=float, default=2.0862256065597453)
    parser.add_argument("--ankle-h2-amp", type=float, default=-0.003434817287963554)
    parser.add_argument("--ankle-h2-phase", type=float, default=1.2927488104774438)
    parser.add_argument("--hip-h3-amp", type=float, default=0.04827596673280693)
    parser.add_argument("--hip-h3-phase", type=float, default=-0.49944083263433436)
    parser.add_argument("--ankle-h3-amp", type=float, default=-0.06968988354403895)
    parser.add_argument("--ankle-h3-phase", type=float, default=1.5873441034476188)
    parser.add_argument("--mpc-horizon", type=int, default=10)
    parser.add_argument("--mpc-candidates", type=int, default=96)
    parser.add_argument("--mpc-cem-iters", type=int, default=0)
    parser.add_argument("--mpc-elite-frac", type=float, default=0.125)
    parser.add_argument("--mpc-num-knots", type=int, default=0)
    parser.add_argument("--mpc-sigma", type=float, default=0.07614211639071694)
    parser.add_argument("--mpc-clip", type=float, default=0.12016284361036686)
    parser.add_argument(
        "--mpc-pose-cost",
        type=float,
        default=23.348190567885954,
    )
    parser.add_argument("--mpc-pitch-target", type=float, default=0.0)
    parser.add_argument("--mpc-yaw-cost", type=float, default=2.7292168081366723)
    parser.add_argument("--mpc-z-cost", type=float, default=2.1215830559511737)
    parser.add_argument(
        "--mpc-z-target",
        type=float,
        default=0.4519975076600261,
    )
    parser.add_argument("--mpc-forward-weight", type=float, default=1.0)
    parser.add_argument("--mpc-ctrl-cost", type=float, default=0.5)
    parser.add_argument(
        "--mpc-terminal-vel-cost",
        type=float,
        default=0.01,
    )
    parser.add_argument("--mpc-plan-decay", type=float, default=0.504186948858276)
    parser.add_argument("--mpc-seed", type=int, default=12)
    parser.add_argument(
        "--mujoco-xml-path",
        type=str,
        default=str(DEFAULT_MUJOCO_XML_PATH),
    )
    parser.add_argument("--trial-name", type=str, default="ant_gait_eval")
    parser.add_argument("--notes", type=str, default="")
    parser.add_argument(
        "--log-path",
        type=str,
        default=str(DEFAULT_LOG_PATH),
    )
    parser.add_argument(
        "--summary-path",
        type=str,
        default=str(DEFAULT_SUMMARY_PATH),
    )
    return parser.parse_args()


if __name__ == "__main__":
    evaluate_heuristic_policy(parse_args())
