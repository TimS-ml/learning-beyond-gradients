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

"""Probe Montezuma's Revenge with a hand-written RAM heuristic.

The current policy focuses on the first room only. It waits for the skull to
reach a configurable horizontal phase, descends the center ladder, performs one
leftward jump macro from the lower middle platform, then executes a configurable
tail macro. The intent is to keep the policy fully heuristic and make each
trial reproducible enough for direct sample-efficiency accounting.

Each invocation appends one JSONL trial record to
`heuristic_montezuma_trials.jsonl` and rewrites
`heuristic_montezuma_trials_summary.csv` with cumulative env-step counts.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from dataclasses import dataclass
from datetime import datetime
from enum import IntEnum
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_LOG_PATH = SCRIPT_DIR / "heuristic_montezuma_trials.jsonl"
DEFAULT_SUMMARY_PATH = SCRIPT_DIR / "heuristic_montezuma_trials_summary.csv"


class MontezumaAction(IntEnum):
    """Discrete actions in MontezumaRevenge's full Atari action space."""

    NOOP = 0
    FIRE = 1
    UP = 2
    RIGHT = 3
    LEFT = 4
    DOWN = 5
    UPRIGHT = 6
    UPLEFT = 7
    DOWNRIGHT = 8
    DOWNLEFT = 9
    UPFIRE = 10
    RIGHTFIRE = 11
    LEFTFIRE = 12
    DOWNFIRE = 13
    UPRIGHTFIRE = 14
    UPLEFTFIRE = 15
    DOWNRIGHTFIRE = 16
    DOWNLEFTFIRE = 17


class PolicyPhase(IntEnum):
    """Finite-state controller phase for one env slot."""

    WAIT_SKULL = 0
    DESCEND_CENTER = 1
    JUMP_LEFT = 2
    TAIL_MACRO = 3
    IDLE = 4


@dataclass(frozen=True)
class MontezumaRamState:
    """Subset of Montezuma RAM bytes used by the hand-written policy."""

    player_x: int
    player_y: int
    motion_state: int
    skull_x: int
    room_id: int
    lives: int | None


@dataclass(frozen=True)
class LeftJumpConfig:
    """Configurable constants for the first-room left-jump heuristic.

    The FSM cycle is:

    1. `WAIT_SKULL`: NOOP until `skull_x <= skull_wait_x` OR
       `wait_timeout_steps` frames elapse. This lets the skull walk past the
       spot the descending player will land on.
    2. `DESCEND_CENTER`: hold DOWN until `player_y >= descend_until_y` AND
       `player_x <= descend_until_x` AND `motion_state == 4` (standing on
       the lower middle platform, not still climbing).
    3. `JUMP_LEFT`: hold `jump_action` (default LEFTFIRE = left + jump) for
       `jump_steps` frames.
    4. `TAIL_MACRO`: hold `tail_action` for `tail_steps` frames as a
       follow-through (grid-searched separately).
    5. `IDLE`: hold `idle_action` forever.

    Life loss immediately jumps to IDLE (see `_handle_life_change`) so the
    agent does not blindly execute a macro after dying.
    """

    skull_wait_x: int = 52
    wait_timeout_steps: int = 256
    descend_until_x: int = 58
    descend_until_y: int = 120
    jump_action: int = MontezumaAction.LEFTFIRE
    jump_steps: int = 14
    tail_action: int = MontezumaAction.NOOP
    tail_steps: int = 24
    idle_action: int = MontezumaAction.NOOP


@dataclass
class PolicyState:
    """Mutable recurrent state for one env slot."""

    phase: PolicyPhase = PolicyPhase.WAIT_SKULL
    phase_steps: int = 0
    episode_steps: int = 0
    last_lives: int | None = None


class FirstRoomLeftJumpAgent:
    """RAM heuristic for one MontezumaRevenge env slot."""

    def __init__(self, config: LeftJumpConfig) -> None:
        self._config = config
        self._state = PolicyState()

    def reset(self) -> None:
        self._state = PolicyState()

    def act(self, ram_state: MontezumaRamState) -> int:
        self._handle_life_change(ram_state.lives)
        action = self._act_for_phase(ram_state)
        self._state.phase_steps += 1
        self._state.episode_steps += 1
        return action

    @property
    def phase(self) -> PolicyPhase:
        return self._state.phase

    def _handle_life_change(self, lives: int | None) -> None:
        if lives is None:
            return
        if self._state.last_lives is not None and lives < self._state.last_lives:
            self._state.phase = PolicyPhase.IDLE
            self._state.phase_steps = 0
        self._state.last_lives = lives

    def _set_phase(self, phase: PolicyPhase) -> None:
        self._state.phase = phase
        self._state.phase_steps = 0

    def _act_for_phase(self, ram_state: MontezumaRamState) -> int:
        if self._state.phase == PolicyPhase.WAIT_SKULL:
            if (
                ram_state.skull_x <= self._config.skull_wait_x
                or self._state.phase_steps >= self._config.wait_timeout_steps
            ):
                self._set_phase(PolicyPhase.DESCEND_CENTER)
                return int(MontezumaAction.DOWN)
            return int(MontezumaAction.NOOP)

        if self._state.phase == PolicyPhase.DESCEND_CENTER:
            if (
                ram_state.player_y >= self._config.descend_until_y
                and ram_state.player_x <= self._config.descend_until_x
                and ram_state.motion_state == 4
            ):
                self._set_phase(PolicyPhase.JUMP_LEFT)
                return int(self._config.jump_action)
            return int(MontezumaAction.DOWN)

        if self._state.phase == PolicyPhase.JUMP_LEFT:
            if self._state.phase_steps >= self._config.jump_steps:
                self._set_phase(PolicyPhase.TAIL_MACRO)
                return int(self._config.tail_action)
            return int(self._config.jump_action)

        if self._state.phase == PolicyPhase.TAIL_MACRO:
            if self._state.phase_steps >= self._config.tail_steps:
                self._set_phase(PolicyPhase.IDLE)
                return int(self._config.idle_action)
            return int(self._config.tail_action)

        return int(self._config.idle_action)


def decode_ram_state(
    info: dict[str, np.ndarray],
    env_index: int,
) -> MontezumaRamState:
    """Decode player/skull coordinates from one env slot's RAM.

    Byte-offset map (empirically established, see docstring in the search
    scripts under `atari/montezuma/` for the full byte survey):

    - `ram[42]` = player x pixel.
    - `ram[43]` = raw player y; the on-screen y is `(312 - raw) & 255`.
    - `ram[2]`  = motion state; 4 means "standing on ground".
    - `ram[47]` = skull x pixel.
    - `ram[3]`  = room id.

    ``lives`` is only present when EnvPool exposes it on the info dict.
    """

    ram = np.asarray(info["ram"])[env_index]
    lives = None
    if "lives" in info:
        lives = int(np.asarray(info["lives"])[env_index])
    return MontezumaRamState(
        player_x=int(ram[42]),
        player_y=(312 - int(ram[43])) & 255,
        motion_state=int(ram[2]),
        skull_x=int(ram[47]),
        room_id=int(ram[3]),
        lives=lives,
    )


def reset_env_with_info(env):
    """Return `(obs, info)` for Gymnasium or `(obs, {})` for legacy Gym."""
    result = env.reset()
    if isinstance(result, tuple):
        return result
    return result, {}


def step_env(env, actions: np.ndarray):
    """Support both legacy Gym and Gymnasium step signatures."""
    result = env.step(actions)
    if len(result) == 5:
        obs, reward, terminated, truncated, info = result
        done = np.logical_or(terminated, truncated)
    else:
        obs, reward, done, info = result
    return obs, np.asarray(reward), np.asarray(done), info


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
    cumulative_ale_frames = 0
    for index, record in enumerate(records):
        env_steps = int(record.get("env_steps", 0))
        ale_frames = int(record.get("ale_frames", 0))
        cumulative_env_steps += env_steps
        cumulative_ale_frames += ale_frames
        rows.append(
            {
                "trial_index": index,
                "timestamp": record.get("timestamp", ""),
                "trial_name": record.get("trial_name", ""),
                "kind": record.get("kind", ""),
                "policy": record.get("policy", ""),
                "num_envs": record.get("num_envs", 1),
                "episodes_started": record.get("episodes_started", 0),
                "episodes_finished": record.get("episodes_finished", 0),
                "env_steps": env_steps,
                "ale_frames": ale_frames,
                "cumulative_env_steps": cumulative_env_steps,
                "cumulative_ale_frames": cumulative_ale_frames,
                "score_mean": record.get("score_mean", ""),
                "score_median": record.get("score_median", ""),
                "score_min": record.get("score_min", ""),
                "score_max": record.get("score_max", ""),
                "min_player_x_alive": min(
                    record.get("min_player_x_alive", [999]),
                ),
                "best_stable_left_x": min(
                    (
                        state["player_x"]
                        for state in record.get(
                            "best_stable_left_states",
                            [],
                        )
                        if state is not None
                    ),
                    default="",
                ),
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
                "policy",
                "num_envs",
                "episodes_started",
                "episodes_finished",
                "env_steps",
                "ale_frames",
                "cumulative_env_steps",
                "cumulative_ale_frames",
                "score_mean",
                "score_median",
                "score_min",
                "score_max",
                "min_player_x_alive",
                "best_stable_left_x",
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
        best_row = max(
            scored_rows,
            key=lambda row: (
                float(row["score_mean"]),
                -float(row["best_stable_left_x"] or 999),
                -float(row["min_player_x_alive"] or 999),
            ),
        )

    print(
        "trial_log_summary:",
        f"records={len(rows)}",
        f"cumulative_env_steps={last_row['cumulative_env_steps']}",
        f"cumulative_ale_frames={last_row['cumulative_ale_frames']}",
    )
    if best_row is not None:
        print(
            "best_mean_score:",
            f"trial_index={best_row['trial_index']}",
            f"trial_name={best_row['trial_name']}",
            f"score_mean={float(best_row['score_mean']):.3f}",
            f"score_median={float(best_row['score_median']):.3f}",
            f"best_stable_left_x={best_row['best_stable_left_x']}",
            f"min_player_x_alive={best_row['min_player_x_alive']}",
            f"cumulative_env_steps={best_row['cumulative_env_steps']}",
        )


def run_one_trial(args: argparse.Namespace, config: LeftJumpConfig) -> None:
    """Evaluate one heuristic config and append a trial record."""
    import envpool

    env = envpool.make_gym(
        "MontezumaRevenge-v5",
        num_envs=args.num_envs,
        batch_size=args.num_envs,
        num_threads=args.num_envs,
        seed=args.seed,
        max_episode_steps=args.max_steps,
        stack_num=1,
        gray_scale=False,
        frame_skip=args.frame_skip,
        noop_max=args.noop_max,
        use_fire_reset=True,
        episodic_life=False,
        reward_clip=False,
        repeat_action_probability=0.0,
        full_action_space=True,
    )

    agents = [FirstRoomLeftJumpAgent(config) for _ in range(args.num_envs)]
    _, info = reset_env_with_info(env)
    for agent in agents:
        agent.reset()
    initial_lives = [
        decode_ram_state(info, env_index).lives
        for env_index in range(args.num_envs)
    ]

    active = np.ones(args.num_envs, dtype=bool)
    episode_scores = np.zeros(args.num_envs, dtype=np.float64)
    episode_lengths = np.zeros(args.num_envs, dtype=np.int64)
    final_states: list[dict[str, Any] | None] = [None] * args.num_envs
    phase_names = [None] * args.num_envs
    min_player_x_alive = np.full(args.num_envs, 999, dtype=np.int64)
    best_stable_left_states: list[dict[str, Any] | None] = [None] * args.num_envs
    env_steps = 0

    for _ in range(args.max_steps):
        if not np.any(active):
            break

        actions = np.zeros(args.num_envs, dtype=np.int32)
        for env_index, agent in enumerate(agents):
            if not active[env_index]:
                actions[env_index] = int(MontezumaAction.NOOP)
                continue
            ram_state = decode_ram_state(info, env_index)
            actions[env_index] = agent.act(ram_state)
            phase_names[env_index] = agent.phase.name.lower()

        _, reward, done, info = step_env(env, actions)
        for env_index in range(args.num_envs):
            if not active[env_index]:
                continue
            episode_scores[env_index] += float(reward[env_index])
            episode_lengths[env_index] += 1
            env_steps += 1
            ram_state = decode_ram_state(info, env_index)
            if (
                initial_lives[env_index] is None
                or ram_state.lives == initial_lives[env_index]
            ):
                min_player_x_alive[env_index] = min(
                    min_player_x_alive[env_index],
                    ram_state.player_x,
                )
            if (
                ram_state.lives == initial_lives[env_index]
                and ram_state.motion_state == 4
                and 100 <= ram_state.player_y <= 140
                and (
                    best_stable_left_states[env_index] is None
                    or ram_state.player_x
                    < best_stable_left_states[env_index]["player_x"]
                )
            ):
                best_stable_left_states[env_index] = {
                    "player_x": ram_state.player_x,
                    "player_y": ram_state.player_y,
                    "motion_state": ram_state.motion_state,
                    "skull_x": ram_state.skull_x,
                    "room_id": ram_state.room_id,
                    "lives": ram_state.lives,
                    "phase": phase_names[env_index],
                    "step": int(episode_lengths[env_index]),
                }
            if bool(done[env_index]):
                active[env_index] = False
                final_states[env_index] = {
                    "player_x": ram_state.player_x,
                    "player_y": ram_state.player_y,
                    "motion_state": ram_state.motion_state,
                    "skull_x": ram_state.skull_x,
                    "room_id": ram_state.room_id,
                    "lives": ram_state.lives,
                    "phase": phase_names[env_index],
                }

    for env_index in range(args.num_envs):
        if final_states[env_index] is None:
            final_state = decode_ram_state(info, env_index)
            final_states[env_index] = {
                "player_x": final_state.player_x,
                "player_y": final_state.player_y,
                "motion_state": final_state.motion_state,
                "skull_x": final_state.skull_x,
                "room_id": final_state.room_id,
                "lives": final_state.lives,
                "phase": phase_names[env_index],
            }

    print(
        "eval_summary:",
        f"num_envs={args.num_envs}",
        f"env_steps={env_steps}",
        f"ale_frames={env_steps * args.frame_skip}",
        f"mean={episode_scores.mean():.3f}",
        f"median={np.median(episode_scores):.3f}",
        f"min={episode_scores.min():.1f}",
        f"max={episode_scores.max():.1f}",
    )
    print("episode_scores:", episode_scores.tolist())
    print("episode_lengths:", episode_lengths.tolist())
    print("final_states:", final_states)
    print("min_player_x_alive:", min_player_x_alive.tolist())
    print("best_stable_left_states:", best_stable_left_states)

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
            "game": "MontezumaRevenge-v5",
            "policy": "first_room_left_jump_ram",
            "num_envs": args.num_envs,
            "seed": args.seed,
            "frame_skip": args.frame_skip,
            "noop_max": args.noop_max,
            "episodes_started": args.num_envs,
            "episodes_finished": args.num_envs,
            "env_steps": env_steps,
            "ale_frames": env_steps * args.frame_skip,
            "score_mean": float(episode_scores.mean()),
            "score_median": float(np.median(episode_scores)),
            "score_min": float(episode_scores.min()),
            "score_max": float(episode_scores.max()),
            "episode_scores": episode_scores.tolist(),
            "episode_lengths": episode_lengths.tolist(),
            "final_states": final_states,
            "min_player_x_alive": min_player_x_alive.tolist(),
            "best_stable_left_states": best_stable_left_states,
            "config": {
                "skull_wait_x": config.skull_wait_x,
                "wait_timeout_steps": config.wait_timeout_steps,
                "descend_until_x": config.descend_until_x,
                "descend_until_y": config.descend_until_y,
                "jump_action": int(config.jump_action),
                "jump_steps": config.jump_steps,
                "tail_action": int(config.tail_action),
                "tail_steps": config.tail_steps,
                "idle_action": int(config.idle_action),
            },
            "notes": args.notes,
        },
    )
    print_summary(write_summary(log_path, summary_path))


def run_grid_search(args: argparse.Namespace) -> None:
    """Scan one left-jump macro grid and log every candidate as one trial."""
    configs = list(
        itertools.product(
            args.grid_skull_wait_x,
            args.grid_descend_until_x,
            args.grid_jump_steps,
            args.grid_tail_action,
            args.grid_tail_steps,
        )
    )
    print(f"grid_size={len(configs)}")
    for index, (
        skull_wait_x,
        descend_until_x,
        jump_steps,
        tail_action,
        tail_steps,
    ) in enumerate(configs):
        config = LeftJumpConfig(
            skull_wait_x=skull_wait_x,
            wait_timeout_steps=args.wait_timeout_steps,
            descend_until_x=descend_until_x,
            descend_until_y=args.descend_until_y,
            jump_action=args.jump_action,
            jump_steps=jump_steps,
            tail_action=tail_action,
            tail_steps=tail_steps,
            idle_action=args.idle_action,
        )
        args.trial_name = (
            f"{args.grid_name}_{index:04d}_"
            f"skull{skull_wait_x}_x{descend_until_x}_"
            f"jump{int(args.jump_action)}x{jump_steps}_"
            f"tail{tail_action}x{tail_steps}"
        )
        run_one_trial(args, config)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a hand-written first-room Montezuma RAM heuristic and "
            "append sample-efficiency logs."
        ),
    )
    parser.add_argument("--trial-name", default="left_jump_probe")
    parser.add_argument(
        "--log-path",
        default=str(DEFAULT_LOG_PATH),
    )
    parser.add_argument(
        "--summary-path",
        default=str(DEFAULT_SUMMARY_PATH),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=20000)
    parser.add_argument("--frame-skip", type=int, default=1)
    parser.add_argument("--noop-max", type=int, default=1)
    parser.add_argument("--notes", default="")
    parser.add_argument("--skull-wait-x", type=int, default=52)
    parser.add_argument("--wait-timeout-steps", type=int, default=256)
    parser.add_argument("--descend-until-x", type=int, default=58)
    parser.add_argument("--descend-until-y", type=int, default=120)
    parser.add_argument(
        "--jump-action",
        type=int,
        default=int(MontezumaAction.LEFTFIRE),
    )
    parser.add_argument("--jump-steps", type=int, default=14)
    parser.add_argument(
        "--tail-action",
        type=int,
        default=int(MontezumaAction.NOOP),
    )
    parser.add_argument("--tail-steps", type=int, default=24)
    parser.add_argument(
        "--idle-action",
        type=int,
        default=int(MontezumaAction.NOOP),
    )
    parser.add_argument(
        "--grid-search",
        action="store_true",
        help="Scan one parameter grid and log every candidate.",
    )
    parser.add_argument("--grid-name", default="left_jump_grid")
    parser.add_argument(
        "--grid-skull-wait-x",
        type=int,
        nargs="+",
        default=[44, 52, 60],
    )
    parser.add_argument(
        "--grid-descend-until-x",
        type=int,
        nargs="+",
        default=[56, 58, 60],
    )
    parser.add_argument(
        "--grid-jump-steps",
        type=int,
        nargs="+",
        default=[10, 12, 14, 16, 18],
    )
    parser.add_argument(
        "--grid-tail-action",
        type=int,
        nargs="+",
        default=[
            int(MontezumaAction.NOOP),
            int(MontezumaAction.LEFT),
            int(MontezumaAction.RIGHT),
            int(MontezumaAction.DOWN),
            int(MontezumaAction.LEFTFIRE),
            int(MontezumaAction.UPLEFTFIRE),
        ],
    )
    parser.add_argument(
        "--grid-tail-steps",
        type=int,
        nargs="+",
        default=[8, 12, 16, 24],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.grid_search:
        run_grid_search(args)
        return
    config = LeftJumpConfig(
        skull_wait_x=args.skull_wait_x,
        wait_timeout_steps=args.wait_timeout_steps,
        descend_until_x=args.descend_until_x,
        descend_until_y=args.descend_until_y,
        jump_action=args.jump_action,
        jump_steps=args.jump_steps,
        tail_action=args.tail_action,
        tail_steps=args.tail_steps,
        idle_action=args.idle_action,
    )
    run_one_trial(args, config)


if __name__ == "__main__":
    main()
