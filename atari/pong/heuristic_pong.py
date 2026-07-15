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

"""Play Atari Pong with a hand-written geometric heuristic.

The policy uses only the current frame plus its own tiny recurrent state:
it segments the ball and paddles from pixels, estimates ball velocity from
consecutive detections, reflects the trajectory against the top/bottom walls,
and moves the right paddle to the predicted interception point.

No frame stacking is required. By default the environment is configured as a
single raw-ish Atari frame (`stack_num=1`, 210x160, grayscale), and the policy
learns which discrete actions mean "up", "down", and "no-op" by probing the
environment for a few reset-rollout steps.

For best scores, use the RAM policy (`--policy ram`, the default). It reads
`info["ram"]`, decodes the ball/paddle coordinates from a few hand-picked
bytes, predicts the next paddle interception, and applies a small impact offset
to steer the outgoing ball away from the opponent.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ActionMap:
    """Discrete action ids used by the right paddle controller."""

    noop: int
    up: int
    down: int


@dataclass(frozen=True)
class PongDetections:
    """Object detections extracted from one Atari Pong frame."""

    ball_xy: tuple[float, float] | None
    self_paddle_y: float | None
    opponent_paddle_y: float | None
    height: int
    width: int

    @property
    def field_top(self) -> float:
        return 0.17 * self.height

    @property
    def field_bottom(self) -> float:
        return 0.93 * self.height

    @property
    def self_paddle_x(self) -> float:
        return 0.92 * self.width

    @property
    def self_contact_x(self) -> float:
        return self.self_paddle_x - 0.03 * self.width

    @property
    def home_y(self) -> float:
        return 105.0


@dataclass(frozen=True)
class PongRamDetections:
    """Object coordinates decoded from Atari RAM."""

    ball_xy: tuple[float, float] | None
    self_paddle_y: float
    opponent_paddle_y: float

    @property
    def field_top(self) -> float:
        return 34.0

    @property
    def field_bottom(self) -> float:
        return 194.0

    @property
    def self_paddle_x(self) -> float:
        return 141.0

    @property
    def home_y(self) -> float:
        return 105.0


@dataclass
class PongPolicyState:
    """Minimal recurrent state maintained by the heuristic policy."""

    prev_ball_xy: tuple[float, float] | None = None
    velocity_xy: tuple[float, float] | None = None
    missing_ball_frames: int = 0


@dataclass(frozen=True)
class HeuristicConfig:
    """Thresholds and control gains for the heuristic agent."""

    ball_threshold: int = 200
    paddle_contrast: int = 30
    paddle_deadband_px: float = 6.0
    spin_offset_px: float = 8.0
    min_paddle_run_px: int = 6
    max_ball_area_px: int = 24
    max_ball_row_px: int = 8
    calibration_steps: int = 6
    calibration_settle_steps: int = 24
    fallback_noop_action: int = 0
    fallback_up_action: int = 2
    fallback_down_action: int = 3
    max_velocity_jump_px: float = 24.0
    max_missing_ball_frames: int = 8


class PongVision:
    """Decode ball and paddle coordinates from one Pong frame."""

    def __init__(self, config: HeuristicConfig) -> None:
        self._config = config

    def detect(self, obs: np.ndarray) -> PongDetections:
        frame = extract_latest_frame(obs)
        luma = to_luma(frame)
        height, width = luma.shape

        field_top = int(round(0.17 * height))
        field_bottom = int(round(0.93 * height))
        ball_top = int(round(0.14 * height))
        ball_bottom = height
        paddle_top = 0
        paddle_bottom = height
        left_x0 = max(0, int(round(0.03 * width)))
        left_x1 = min(width, int(round(0.18 * width)))
        right_x0 = max(0, int(round(0.82 * width)))
        right_x1 = min(width, int(round(0.98 * width)))
        ball_x0 = max(0, int(round(0.14 * width)))
        ball_x1 = min(width, int(round(0.98 * width)))
        net_x0 = max(ball_x0, int(round(0.49 * width)))
        net_x1 = min(ball_x1, int(round(0.52 * width)))

        self_paddle_y = self._detect_paddle_y(
            luma,
            x0=right_x0,
            x1=right_x1,
            y0=paddle_top,
            y1=paddle_bottom,
        )
        opponent_paddle_y = self._detect_paddle_y(
            luma,
            x0=left_x0,
            x1=left_x1,
            y0=paddle_top,
            y1=paddle_bottom,
        )
        ball_xy = self._detect_ball_xy(
            luma,
            x0=ball_x0,
            x1=ball_x1,
            y0=ball_top,
            y1=ball_bottom,
            net_x0=net_x0,
            net_x1=net_x1,
        )
        return PongDetections(
            ball_xy=ball_xy,
            self_paddle_y=self_paddle_y,
            opponent_paddle_y=opponent_paddle_y,
            height=height,
            width=width,
        )

    def _detect_paddle_y(
        self,
        luma: np.ndarray,
        *,
        x0: int,
        x1: int,
        y0: int,
        y1: int,
    ) -> float | None:
        zone = luma[y0:y1, x0:x1]
        background = dominant_pixel_value(zone)
        object_mask = (
            np.abs(zone.astype(np.int16) - background)
            >= self._config.paddle_contrast
        )
        row_pixel_count = np.count_nonzero(object_mask, axis=1)
        max_paddle_width = max(2, (x1 - x0) // 2)
        row_has_object = (row_pixel_count >= 2) & (
            row_pixel_count <= max_paddle_width
        )
        center = longest_true_run_center(
            row_has_object, min_len=self._config.min_paddle_run_px
        )
        if center is None and self._config.min_paddle_run_px > 2:
            center = longest_true_run_center(row_has_object, min_len=2)
        if center is None:
            return None
        return float(y0 + center)

    def _detect_ball_xy(
        self,
        luma: np.ndarray,
        *,
        x0: int,
        x1: int,
        y0: int,
        y1: int,
        net_x0: int,
        net_x1: int,
    ) -> tuple[float, float] | None:
        mask = luma[y0:y1, x0:x1] >= self._config.ball_threshold
        mask[:, max(0, net_x0 - x0) : max(0, net_x1 - x0)] = False
        mask[np.count_nonzero(mask, axis=1) > self._config.max_ball_row_px, :] = False
        components = connected_components(mask)
        best_component: np.ndarray | None = None
        best_area = self._config.max_ball_area_px + 1
        for component in components:
            area = component.shape[0]
            if area > self._config.max_ball_area_px:
                continue
            span_y = component[:, 0].max() - component[:, 0].min() + 1
            span_x = component[:, 1].max() - component[:, 1].min() + 1
            if span_y > 8 or span_x > 8:
                continue
            if area < best_area:
                best_component = component
                best_area = area
        if best_component is None:
            return None
        center_yx = best_component.mean(axis=0)
        return float(x0 + center_yx[1]), float(y0 + center_yx[0])


class HeuristicPongAgent:
    """Geometry-only Pong controller with trajectory prediction."""

    def __init__(
        self,
        action_map: ActionMap,
        vision: PongVision,
        config: HeuristicConfig,
    ) -> None:
        self._action_map = action_map
        self._vision = vision
        self._config = config
        self._state = PongPolicyState()

    def reset(self) -> None:
        self._state = PongPolicyState()

    def act(self, obs: np.ndarray) -> int:
        detections = self._vision.detect(obs)
        self._update_ball_state(detections.ball_xy)

        target_y = detections.home_y
        if detections.self_paddle_y is None:
            return self._action_map.noop

        if detections.ball_xy is not None:
            ball_x, ball_y = detections.ball_xy
            velocity = self._state.velocity_xy
            if velocity is not None and velocity[0] > 0.1:
                if ball_x <= detections.self_contact_x:
                    intercept_y = self._predict_intercept_y(
                        detections,
                        velocity,
                    )
                    target_y = self._apply_spin_bias(
                        intercept_y,
                        detections.opponent_paddle_y,
                        detections.home_y,
                    )
                else:
                    target_y = self._apply_spin_bias(
                        ball_y,
                        detections.opponent_paddle_y,
                        detections.home_y,
                    )
            elif velocity is None and ball_x >= 0.55 * detections.width:
                target_y = ball_y

        error = target_y - detections.self_paddle_y
        if error < -self._config.paddle_deadband_px:
            return self._action_map.up
        if error > self._config.paddle_deadband_px:
            return self._action_map.down
        return self._action_map.noop

    def _update_ball_state(
        self, ball_xy: tuple[float, float] | None
    ) -> None:
        if ball_xy is None:
            self._state.missing_ball_frames += 1
            if (
                self._state.missing_ball_frames
                > self._config.max_missing_ball_frames
            ):
                self._state.prev_ball_xy = None
                self._state.velocity_xy = None
            return

        self._state.missing_ball_frames = 0
        if self._state.prev_ball_xy is not None:
            dx = ball_xy[0] - self._state.prev_ball_xy[0]
            dy = ball_xy[1] - self._state.prev_ball_xy[1]
            if (
                abs(dx) <= self._config.max_velocity_jump_px
                and abs(dy) <= self._config.max_velocity_jump_px
                and abs(dx) + abs(dy) > 0.25
            ):
                if self._state.velocity_xy is None:
                    self._state.velocity_xy = (dx, dy)
                else:
                    old_dx, old_dy = self._state.velocity_xy
                    self._state.velocity_xy = (
                        0.5 * old_dx + 0.5 * dx,
                        0.5 * old_dy + 0.5 * dy,
                    )
            else:
                self._state.velocity_xy = None
        self._state.prev_ball_xy = ball_xy

    def _predict_intercept_y(
        self, detections: PongDetections, velocity_xy: tuple[float, float]
    ) -> float:
        if detections.ball_xy is None:
            return detections.home_y
        ball_x, ball_y = detections.ball_xy
        vx, vy = velocity_xy
        if vx <= 0.1:
            return ball_y
        impact_x = detections.self_contact_x
        raw_y = ball_y + vy / vx * (impact_x - ball_x)
        return reflect_position(
            raw_y,
            lower=detections.field_top + 2.0,
            upper=detections.field_bottom - 2.0,
        )

    def _apply_spin_bias(
        self,
        intercept_y: float,
        opponent_paddle_y: float | None,
        home_y: float,
    ) -> float:
        # Key insight: Atari Pong's paddle spins the ball based on which part
        # of the paddle makes contact. Aiming for the top of our paddle sends
        # the return down; aiming for the bottom sends it up. So if the
        # opponent is above center, bias contact toward the top of our paddle
        # (subtract offset) to make the return travel downward past them,
        # and vice versa. This is what lifts scores from "reliable rally" to
        # "reliable 21".
        if opponent_paddle_y is None:
            return intercept_y
        desired_outgoing_sign = 1.0 if opponent_paddle_y < home_y else -1.0
        return intercept_y - desired_outgoing_sign * self._config.spin_offset_px


class RamPongAgent:
    """RAM-based Pong controller with the same geometric policy."""

    def __init__(
        self,
        action_map: ActionMap,
        config: HeuristicConfig,
    ) -> None:
        self._action_map = action_map
        self._config = config
        self._state = PongPolicyState()

    def reset(self) -> None:
        self._state = PongPolicyState()

    def act(self, info: dict[str, np.ndarray]) -> int:
        detections = decode_ram_detections(info)
        self._update_ball_state(detections.ball_xy)

        target_y = detections.home_y
        velocity = self._state.velocity_xy
        if detections.ball_xy is not None:
            ball_x, ball_y = detections.ball_xy
            if velocity is not None and velocity[0] > 0.05:
                if ball_x <= detections.self_paddle_x:
                    intercept_y = reflect_position(
                        ball_y
                        + velocity[1]
                        / velocity[0]
                        * (detections.self_paddle_x - ball_x),
                        lower=detections.field_top,
                        upper=detections.field_bottom,
                    )
                    target_y = self._apply_spin_bias(
                        intercept_y,
                        detections.opponent_paddle_y,
                        detections.home_y,
                    )
                else:
                    target_y = self._apply_spin_bias(
                        ball_y,
                        detections.opponent_paddle_y,
                        detections.home_y,
                    )
            elif velocity is None and ball_x > 90.0:
                target_y = ball_y

        error = target_y - detections.self_paddle_y
        if error < -self._config.paddle_deadband_px:
            return self._action_map.up
        if error > self._config.paddle_deadband_px:
            return self._action_map.down
        return self._action_map.noop

    def _update_ball_state(
        self, ball_xy: tuple[float, float] | None
    ) -> None:
        if ball_xy is None:
            self._state.prev_ball_xy = None
            self._state.velocity_xy = None
            return

        if self._state.prev_ball_xy is not None:
            dx = ball_xy[0] - self._state.prev_ball_xy[0]
            dy = ball_xy[1] - self._state.prev_ball_xy[1]
            if (
                abs(dx) <= self._config.max_velocity_jump_px
                and abs(dy) <= self._config.max_velocity_jump_px
                and abs(dx) + abs(dy) > 0.25
            ):
                if self._state.velocity_xy is None:
                    self._state.velocity_xy = (dx, dy)
                else:
                    old_dx, old_dy = self._state.velocity_xy
                    self._state.velocity_xy = (
                        0.5 * old_dx + 0.5 * dx,
                        0.5 * old_dy + 0.5 * dy,
                    )
            else:
                self._state.velocity_xy = None
        self._state.prev_ball_xy = ball_xy

    def _apply_spin_bias(
        self,
        intercept_y: float,
        opponent_paddle_y: float,
        home_y: float,
    ) -> float:
        desired_outgoing_sign = 1.0 if opponent_paddle_y < home_y else -1.0
        return intercept_y - desired_outgoing_sign * self._config.spin_offset_px


def extract_latest_frame(obs: np.ndarray) -> np.ndarray:
    """Return the newest per-env frame as HxW or HxWx3/4."""
    arr = np.asarray(obs)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim == 2:
        return arr
    if arr.ndim != 3:
        raise ValueError(f"Unsupported observation shape: {arr.shape}")

    channel_first = arr.shape[0] <= 12 and arr.shape[1] > 16 and arr.shape[2] > 16
    if channel_first:
        channels = arr.shape[0]
        if channels == 1:
            return arr[0]
        if channels % 3 == 0:
            return np.moveaxis(arr[-3:], 0, -1)
        return arr[-1]
    if arr.shape[-1] == 1:
        return arr[..., 0]
    return arr[..., :3]


def to_luma(frame: np.ndarray) -> np.ndarray:
    """Convert a frame to uint8 luminance."""
    arr = np.asarray(frame)
    if arr.ndim == 2:
        return arr.astype(np.uint8, copy=False)
    rgb = arr[..., :3].astype(np.uint16, copy=False)
    luma = (77 * rgb[..., 0] + 150 * rgb[..., 1] + 29 * rgb[..., 2]) >> 8
    return luma.astype(np.uint8, copy=False)


def longest_true_run_center(mask: np.ndarray, min_len: int) -> float | None:
    """Center index of the longest contiguous true run."""
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        return None
    breaks = np.flatnonzero(np.diff(indices) > 1)
    start_offsets = np.concatenate(([0], breaks + 1))
    end_offsets = np.concatenate((breaks, [indices.size - 1]))
    best_len = 0
    best_center = None
    for start_offset, end_offset in zip(start_offsets, end_offsets, strict=True):
        run_start = indices[start_offset]
        run_end = indices[end_offset]
        run_len = run_end - run_start + 1
        if run_len >= min_len and run_len > best_len:
            best_len = run_len
            best_center = 0.5 * (run_start + run_end)
    return best_center


def connected_components(mask: np.ndarray) -> list[np.ndarray]:
    """Connected components over true pixels in a tiny binary mask."""
    coords = np.argwhere(mask)
    if coords.size == 0:
        return []
    index_by_coord = {tuple(coord): i for i, coord in enumerate(coords.tolist())}
    visited = np.zeros(coords.shape[0], dtype=bool)
    components: list[np.ndarray] = []

    for root_idx in range(coords.shape[0]):
        if visited[root_idx]:
            continue
        visited[root_idx] = True
        stack = [root_idx]
        component_indices = [root_idx]
        while stack:
            idx = stack.pop()
            y, x = coords[idx]
            for ny, nx in (
                (y - 1, x),
                (y + 1, x),
                (y, x - 1),
                (y, x + 1),
            ):
                next_idx = index_by_coord.get((ny, nx))
                if next_idx is None or visited[next_idx]:
                    continue
                visited[next_idx] = True
                stack.append(next_idx)
                component_indices.append(next_idx)
        components.append(coords[np.asarray(component_indices)])
    return components


def dominant_pixel_value(frame: np.ndarray) -> int:
    """Most frequent uint8 value in a tiny ROI."""
    values, counts = np.unique(frame, return_counts=True)
    return int(values[int(np.argmax(counts))])


def reflect_position(value: float, *, lower: float, upper: float) -> float:
    """Reflect a 1D coordinate inside [lower, upper] with elastic walls."""
    span = upper - lower
    if span <= 0:
        return lower
    period = 2.0 * span
    shifted = (value - lower) % period
    if shifted <= span:
        return lower + shifted
    return upper - (shifted - span)


def decode_ram_detections(info: dict[str, np.ndarray]) -> PongRamDetections:
    """Decode one-env Pong coordinates from `info["ram"]`."""
    ram = np.asarray(info["ram"])[0]
    ball_xy = None
    if int(ram[54]) != 0:
        ball_xy = (float(ram[49]) - 49.0, float(ram[54]) - 13.0)
    return PongRamDetections(
        ball_xy=ball_xy,
        self_paddle_y=0.972157 * float(ram[51]) - 2.553996,
        opponent_paddle_y=0.981619 * float(ram[50]) - 5.492890,
    )


def reset_env(env):
    """Support both legacy Gym and Gymnasium reset signatures."""
    result = env.reset()
    if isinstance(result, tuple):
        return result[0]
    return result


def reset_env_with_info(env):
    """Return `(obs, info)` for Gymnasium or `(obs, {})` for legacy Gym."""
    result = env.reset()
    if isinstance(result, tuple):
        return result
    return result, {}


def step_env(env, action: int):
    """Support both legacy Gym and Gymnasium step signatures."""
    result = env.step(np.asarray([action], dtype=np.int32))
    if len(result) == 5:
        obs, reward, terminated, truncated, info = result
        done = np.logical_or(terminated, truncated)
    else:
        obs, reward, done, info = result
    return obs, float(np.asarray(reward)[0]), bool(np.asarray(done)[0]), info


def calibrate_action_map(
    env,
    vision: PongVision,
    config: HeuristicConfig,
) -> ActionMap:
    """Probe the environment to discover no-op/up/down action ids.

    Instead of hardcoding action ids per ROM variant, run each discrete
    action for a few frames from a fresh reset, measure the paddle's Y
    displacement using the vision detector, and label:

    - ``up`` = action with the most negative Y delta,
    - ``down`` = action with the most positive Y delta,
    - ``noop`` = action with the smallest absolute delta.

    Falls back to ``fallback_*_action`` from the config if the vision
    detector never latched on to the paddle (all deltas below 1 pixel). This
    is one of the reusable "figure out what actions mean by trying them"
    patterns the blog names as a heuristic building block.
    """

    action_num = int(env.action_space.n)
    mean_deltas: list[float] = []
    for action in range(action_num):
        obs = reset_env(env)
        for _ in range(config.calibration_settle_steps):
            if vision.detect(obs).self_paddle_y is not None:
                break
            obs, _, done, _ = step_env(env, config.fallback_noop_action)
            if done:
                obs = reset_env(env)
        start_y = vision.detect(obs).self_paddle_y
        end_y = start_y
        for _ in range(config.calibration_steps):
            obs, _, done, _ = step_env(env, action)
            detected_y = vision.detect(obs).self_paddle_y
            if detected_y is not None:
                end_y = detected_y
            if done:
                break
        if start_y is None or end_y is None:
            mean_deltas.append(0.0)
        else:
            mean_deltas.append(end_y - start_y)

    deltas = np.asarray(mean_deltas, dtype=np.float32)
    up = int(np.argmin(deltas))
    down = int(np.argmax(deltas))
    noop = int(np.argmin(np.abs(deltas)))

    if abs(float(deltas[up])) < 1.0 or abs(float(deltas[down])) < 1.0:
        return ActionMap(
            noop=config.fallback_noop_action,
            up=config.fallback_up_action,
            down=config.fallback_down_action,
        )
    return ActionMap(noop=noop, up=up, down=down)


def evaluate_heuristic_policy(args: argparse.Namespace) -> None:
    """Run the heuristic policy and print per-episode Pong scores."""
    import envpool

    frame_skip = args.frame_skip
    if frame_skip is None:
        frame_skip = 1 if args.policy == "ram" else 2

    env = envpool.make_gym(
        "Pong-v5",
        num_envs=1,
        batch_size=1,
        seed=args.seed,
        img_height=args.img_height,
        img_width=args.img_width,
        stack_num=args.stack_num,
        gray_scale=args.gray_scale,
        frame_skip=frame_skip,
        noop_max=args.noop_max,
        use_fire_reset=not args.disable_fire_reset,
        episodic_life=False,
        reward_clip=False,
        repeat_action_probability=0.0,
        full_action_space=False,
    )

    config = HeuristicConfig(
        ball_threshold=args.ball_threshold,
        paddle_contrast=args.paddle_contrast,
        paddle_deadband_px=args.deadband,
        spin_offset_px=args.spin_offset,
        calibration_steps=args.calibration_steps,
        calibration_settle_steps=args.calibration_settle_steps,
    )
    vision = PongVision(config)
    if args.policy == "ram":
        action_map = ActionMap(
            noop=args.noop_action,
            up=args.up_action,
            down=args.down_action,
        )
        agent = RamPongAgent(action_map=action_map, config=config)
    elif args.auto_calibrate_actions:
        action_map = calibrate_action_map(env, vision, config)
        print(
            "calibrated actions:",
            f"noop={action_map.noop}",
            f"up={action_map.up}",
            f"down={action_map.down}",
        )
        agent = HeuristicPongAgent(
            action_map=action_map,
            vision=vision,
            config=config,
        )
    else:
        action_map = ActionMap(
            noop=args.noop_action,
            up=args.up_action,
            down=args.down_action,
        )
        agent = HeuristicPongAgent(
            action_map=action_map,
            vision=vision,
            config=config,
        )

    scores = []
    for episode in range(args.episodes):
        obs, info = reset_env_with_info(env)
        agent.reset()
        total_reward = 0.0
        for _ in range(args.max_steps):
            action = agent.act(info if args.policy == "ram" else obs)
            obs, reward, done, info = step_env(env, action)
            total_reward += reward
            if done:
                break
        scores.append(total_reward)
        print(f"episode={episode} score={total_reward:.1f}")

    score_arr = np.asarray(scores, dtype=np.float32)
    print(
        "summary:",
        f"episodes={len(scores)}",
        f"mean={score_arr.mean():.3f}",
        f"min={score_arr.min():.1f}",
        f"max={score_arr.max():.1f}",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a pure heuristic Atari Pong policy."
    )
    parser.add_argument(
        "--policy",
        choices=("ram", "vision"),
        default="ram",
        help="Use RAM decoding for best score, or vision for pixel debugging.",
    )
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=27000)
    parser.add_argument("--img-height", type=int, default=210)
    parser.add_argument("--img-width", type=int, default=160)
    parser.add_argument("--stack-num", type=int, default=1)
    parser.add_argument(
        "--gray-scale",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=None,
        help="Default: 1 for RAM policy, 2 for vision policy.",
    )
    parser.add_argument("--noop-max", type=int, default=1)
    parser.add_argument("--disable-fire-reset", action="store_true")
    parser.add_argument("--ball-threshold", type=int, default=200)
    parser.add_argument("--paddle-contrast", type=int, default=30)
    parser.add_argument("--deadband", type=float, default=6.0)
    parser.add_argument("--spin-offset", type=float, default=8.0)
    parser.add_argument("--calibration-steps", type=int, default=6)
    parser.add_argument("--calibration-settle-steps", type=int, default=24)
    parser.add_argument(
        "--auto-calibrate-actions",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--noop-action", type=int, default=0)
    parser.add_argument("--up-action", type=int, default=2)
    parser.add_argument("--down-action", type=int, default=3)
    return parser.parse_args()


if __name__ == "__main__":
    evaluate_heuristic_policy(parse_args())
