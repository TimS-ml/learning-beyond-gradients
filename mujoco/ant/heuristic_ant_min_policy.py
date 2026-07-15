"""Minimal stateful Ant-v5 policy extracted from the current best MPC heuristic.

External API:
  policy = AntMPCPolicy()
  policy.reset()
  action = policy.act(obs)  # obs is one env's observation array

The policy is not a pure stateless `f(obs)` because it keeps an oscillator
phase and a warm-started MPC residual plan across steps.
"""

from __future__ import annotations

import math
from pathlib import Path

import mujoco
import numpy as np


XML_PATH = Path(__file__).resolve().parent / "ant_envpool.xml"

Q_INDEX = np.asarray([6, 7, 0, 1, 2, 3, 4, 5], dtype=np.int64)
LEG_PHASE = np.asarray([0.0, math.pi, 0.0, math.pi], dtype=np.float64)
HIP_SIGN = np.asarray([-1.0, 1.0, 1.0, -1.0], dtype=np.float64)
ANKLE_SIGN = np.asarray([1.0, 1.0, -1.0, -1.0], dtype=np.float64)
HEADING_AXIS = np.asarray([1.0, 1.0, -1.0, -1.0], dtype=np.float64)
PITCH_AXIS = np.asarray([1.0, 1.0, -1.0, -1.0], dtype=np.float64)
ROLL_AXIS = np.asarray([-1.0, 1.0, 1.0, -1.0], dtype=np.float64)
FOOT_ROWS = np.asarray([12, 3, 6, 9], dtype=np.int64)
FOOT_BODY_IDS = np.asarray([13, 4, 7, 10], dtype=np.int64)

D_PHI = 0.660934259732249
D_PHI_SPEED_GAIN = -0.02
D_PHI_SPEED_TARGET = 5.8
D_PHI_MIN = 0.62
D_PHI_MAX = 0.72

HIP_BIAS = 0.12217781430672398
HIP_AMP = 0.5705418365199333
HIP_H2_AMP = 0.10975404801587477
HIP_H2_PHASE = 2.0862256065597453
HIP_H3_AMP = 0.04827596673280693
HIP_H3_PHASE = -0.49944083263433436
HIP_STANCE_SCALE = 1.0479076970107701
HIP_SWING_SCALE = 1.0031777685985328

ANKLE_BIAS = 0.36651046903486795
ANKLE_AMP = 0.26587749767314783
ANKLE_H2_AMP = -0.003434817287963554
ANKLE_H2_PHASE = 1.2927488104774438
ANKLE_H3_AMP = -0.06968988354403895
ANKLE_H3_PHASE = 1.5873441034476188
ANKLE_STANCE_SCALE = 0.976603459922793
ANKLE_SWING_SCALE = 0.9374473230114526

STANCE_DUTY = 0.6355364206196007
STANCE_DUTY_SPEED_GAIN = -0.01
STANCE_DUTY_SPEED_TARGET = 5.8
STANCE_DUTY_MIN = 0.6
STANCE_DUTY_MAX = 0.67

KP = 0.8108143632989734
PITCH_GAIN = -0.19444590796726124
PITCH_RATE_GAIN = 0.04099276700871415
ROLL_GAIN = -0.25536960225655303
ROLL_RATE_GAIN = 0.023293075237761272
YAW_GAIN = -0.12067720879887742
YAW_RATE_GAIN = 0.04418873596679619

HORIZON = 10
CANDIDATES = 96
MPC_SIGMA = 0.07614211639071694
MPC_CLIP = 0.12016284361036686
POSE_COST = 23.348190567885954
YAW_COST = 2.7292168081366723
Z_COST = 2.1215830559511737
Z_TARGET = 0.4519975076600261
TERMINAL_VEL_COST = 0.01
PLAN_DECAY = 0.504186948858276
MPC_SEED = 12


def _flatten_obs(obs: np.ndarray) -> np.ndarray:
    obs = np.asarray(obs, dtype=np.float64)
    if obs.ndim == 2:
        obs = obs[0]
    if obs.ndim != 1 or obs.shape[0] < 27:
        raise ValueError(f"Unsupported Ant obs shape: {obs.shape}")
    return obs


def _quat_to_rpy(quat: np.ndarray) -> tuple[float, float, float]:
    w, x, y, z = quat / (np.linalg.norm(quat) + 1e-12)
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw


def _stance_duty(vx: float) -> float:
    return float(
        np.clip(
            STANCE_DUTY
            + STANCE_DUTY_SPEED_GAIN * (vx - STANCE_DUTY_SPEED_TARGET),
            STANCE_DUTY_MIN,
            STANCE_DUTY_MAX,
        )
    )


def _dphi(vx: float) -> float:
    return float(
        np.clip(
            D_PHI + D_PHI_SPEED_GAIN * (vx - D_PHI_SPEED_TARGET),
            D_PHI_MIN,
            D_PHI_MAX,
        )
    )


def _warp_phase(phase: np.ndarray, duty: float) -> np.ndarray:
    phase01 = np.mod(phase, 2.0 * math.pi) / (2.0 * math.pi)
    warped01 = np.where(
        phase01 < duty,
        0.5 * phase01 / duty,
        0.5 + 0.5 * (phase01 - duty) / (1.0 - duty),
    )
    return 2.0 * math.pi * warped01


def _foot_contacts(obs: np.ndarray) -> np.ndarray:
    if obs.shape[0] < 27 + 13 * 6:
        return np.zeros(4, dtype=np.float64)
    return np.clip(obs[27 : 27 + 13 * 6].reshape(13, 6)[FOOT_ROWS, 5], 0.0, 1.0)


def _cpg_action(
    phase: float,
    q: np.ndarray,
    dq: np.ndarray,
    roll: float,
    pitch: float,
    yaw: float,
    roll_rate: float,
    pitch_rate: float,
    yaw_rate: float,
    contacts: np.ndarray,
    vx: float,
) -> np.ndarray:
    leg_phase = _warp_phase(phase + LEG_PHASE, _stance_duty(vx))
    stance = leg_phase < math.pi

    hip_wave = HIP_BIAS + np.where(stance, HIP_STANCE_SCALE, HIP_SWING_SCALE) * (
        HIP_AMP * np.sin(leg_phase)
        + HIP_H2_AMP * np.sin(2.0 * leg_phase + HIP_H2_PHASE)
        + HIP_H3_AMP * np.sin(3.0 * leg_phase + HIP_H3_PHASE)
    )
    ankle_wave = ANKLE_BIAS + np.where(
        stance,
        ANKLE_STANCE_SCALE,
        ANKLE_SWING_SCALE,
    ) * (
        ANKLE_AMP * np.cos(leg_phase)
        + ANKLE_H2_AMP * np.cos(2.0 * leg_phase + ANKLE_H2_PHASE)
        + ANKLE_H3_AMP * np.cos(3.0 * leg_phase + ANKLE_H3_PHASE)
    )

    balance = (
        PITCH_AXIS * (PITCH_GAIN * pitch + PITCH_RATE_GAIN * pitch_rate)
        - ROLL_AXIS * (ROLL_GAIN * roll + ROLL_RATE_GAIN * roll_rate)
    )

    action = np.empty(8, dtype=np.float64)
    action[0::2] = KP * (
        HIP_SIGN * hip_wave
        + HEADING_AXIS * (YAW_GAIN * yaw + YAW_RATE_GAIN * yaw_rate)
        - q[0::2]
    )
    action[1::2] = KP * (ANKLE_SIGN * (ankle_wave + balance) - q[1::2])
    return np.clip(action, -1.0, 1.0)


class AntMPCPolicy:
    """Minimal embeddable Ant-v5 policy: CPG base + residual random-shooting MPC.

    This is the same policy as `heuristic_ant.py --policy mpc`, stripped of
    the CLI, JSONL trial log, and CEM path. It owns:

    - A private `mujoco.MjModel` / `MjData` pair for online rollouts
      (`self.model`, `self.data`).
    - A phase scalar advanced by `_dphi(vx)` every real env step.
    - A warm-started residual plan `self.plan` of shape `(HORIZON, 8)`.
    - A deterministic `numpy.random.Generator` seeded by `MPC_SEED`.

    Exposes only `reset()` and `act(obs) -> action` so it can be imported by
    another project (`from heuristic_ant_min_policy import AntMPCPolicy`).
    """

    def __init__(self) -> None:
        self.model = mujoco.MjModel.from_xml_path(str(XML_PATH))
        self.data = mujoco.MjData(self.model)
        self.phase = 0.0
        self.plan = np.zeros((HORIZON, 8), dtype=np.float64)
        self.rng = np.random.default_rng(MPC_SEED)

    def reset(self) -> None:
        self.phase = 0.0
        self.plan.fill(0.0)
        # Reset the RNG to the fixed seed so two `reset()` calls from the
        # same policy produce identical action trajectories.
        self.rng = np.random.default_rng(MPC_SEED)

    def act(self, obs: np.ndarray) -> np.ndarray:
        obs = _flatten_obs(obs)
        self._set_state_from_obs(obs)
        q = self.data.qpos[7:15][Q_INDEX]
        dq = self.data.qvel[6:14][Q_INDEX]
        vx = float(obs[13])
        roll, pitch, yaw = _quat_to_rpy(obs[1:5])
        roll_rate, pitch_rate, yaw_rate = obs[16:19]

        base = _cpg_action(
            self.phase,
            q,
            dq,
            roll,
            pitch,
            yaw,
            float(roll_rate),
            float(pitch_rate),
            float(yaw_rate),
            _foot_contacts(obs),
            vx,
        )

        best_plan = self.plan.copy()
        best_obj = self._rollout_objective(obs, best_plan)
        for _ in range(CANDIDATES - 1):
            residuals = np.clip(
                best_plan
                + self.rng.normal(0.0, MPC_SIGMA, size=(HORIZON, 8)),
                -MPC_CLIP,
                MPC_CLIP,
            )
            residuals[1:] = 0.6 * residuals[1:] + 0.4 * residuals[:-1]
            obj = self._rollout_objective(obs, residuals)
            if obj > best_obj:
                best_obj = obj
                best_plan = residuals

        self.phase += _dphi(vx)
        self.plan[:-1] = PLAN_DECAY * best_plan[1:]
        self.plan[-1] = 0.0
        return np.clip(base + best_plan[0], -1.0, 1.0)

    def _set_state_from_obs(self, obs: np.ndarray) -> None:
        self.data.qpos[0:2] = 0.0
        self.data.qpos[2:] = obs[:13]
        self.data.qvel[:] = obs[13:27]
        mujoco.mj_forward(self.model, self.data)

    def _rollout_objective(self, obs: np.ndarray, residuals: np.ndarray) -> float:
        self._set_state_from_obs(obs)
        phase = self.phase
        total = 0.0

        for step_idx, residual in enumerate(residuals):
            q = self.data.qpos[7:15][Q_INDEX]
            dq = self.data.qvel[6:14][Q_INDEX]
            vx_before = float(self.data.qvel[0])
            roll, pitch, yaw = _quat_to_rpy(self.data.qpos[3:7])
            roll_rate, pitch_rate, yaw_rate = self.data.qvel[3:6]
            contacts = (
                _foot_contacts(obs)
                if step_idx == 0
                else np.clip(self.data.cfrc_ext[FOOT_BODY_IDS, 5], 0.0, 1.0)
            )

            action = np.clip(
                _cpg_action(
                    phase,
                    q,
                    dq,
                    roll,
                    pitch,
                    yaw,
                    float(roll_rate),
                    float(pitch_rate),
                    float(yaw_rate),
                    contacts,
                    vx_before,
                )
                + residual,
                -1.0,
                1.0,
            )

            x_before = float(self.data.qpos[0])
            self.data.ctrl[:] = action
            for _ in range(5):
                mujoco.mj_step(self.model, self.data)

            vx = (float(self.data.qpos[0]) - x_before) / (
                5.0 * self.model.opt.timestep
            )
            z = float(self.data.qpos[2])
            roll, pitch, yaw = _quat_to_rpy(self.data.qpos[3:7])

            total += vx + (1.0 if 0.2 <= z <= 1.0 else -50.0)
            total -= 0.5 * float(np.square(action).sum())
            total -= POSE_COST * (roll * roll + pitch * pitch)
            total -= YAW_COST * yaw * yaw
            total -= Z_COST * (z - Z_TARGET) ** 2
            if z < 0.23 or z > 0.95:
                total -= 100.0

            phase += _dphi(vx)

        qvel = self.data.qvel[6:14]
        total -= TERMINAL_VEL_COST * float(np.dot(qvel, qvel))
        return total


if __name__ == "__main__":
    import envpool

    env = envpool.make_gym(
        "Ant-v5",
        num_envs=1,
        batch_size=1,
        seed=0,
        max_episode_steps=1000,
    )
    policy = AntMPCPolicy()
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]
    policy.reset()

    score = 0.0
    x_position = 0.0
    for step in range(1, 1001):
        result = env.step(policy.act(obs)[None, :])
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            done = np.logical_or(terminated, truncated)
        else:
            obs, reward, done, info = result
        score += float(np.asarray(reward)[0])
        x_position = float(np.asarray(info["x_position"])[0])
        if bool(np.asarray(done)[0]):
            break

    env.close()
    print(f"score={score:.3f} x_position={x_position:.3f} steps={step}")
