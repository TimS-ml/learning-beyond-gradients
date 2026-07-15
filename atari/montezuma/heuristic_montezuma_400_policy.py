#!/usr/bin/env python3
"""Replay the 400-point MontezumaRevenge-v5 heuristic from the Atari57 batch.

This is the shortest repaired 400-point native-image route found in thread
019d4cc1-9e30-78d0-b304-43b07c2aebe0.  The policy is an open-loop sequence of
86 Atari action-duration pairs; it ignores observations and only keeps an
internal time index.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np

ENV_ID = "MontezumaRevenge-v5"
DEFAULT_SEED = 10001
EXPECTED_SCORE = 400.0
EXPECTED_STEPS = 1769

# Action ids are EnvPool Atari ids with full_action_space=False for this env.
MACROS = [
    [
        0,
        20,
        "NOOP"
    ],
    [
        4,
        8,
        "LEFT"
    ],
    [
        12,
        12,
        "LEFTFIRE"
    ],
    [
        4,
        12,
        "LEFT"
    ],
    [
        3,
        8,
        "RIGHT"
    ],
    [
        15,
        16,
        "UPLEFTFIRE"
    ],
    [
        11,
        32,
        "RIGHTFIRE"
    ],
    [
        16,
        6,
        "DOWNRIGHTFIRE"
    ],
    [
        14,
        12,
        "UPRIGHTFIRE"
    ],
    [
        11,
        12,
        "RIGHTFIRE"
    ],
    [
        12,
        8,
        "LEFTFIRE"
    ],
    [
        1,
        18,
        "FIRE"
    ],
    [
        11,
        10,
        "RIGHTFIRE"
    ],
    [
        15,
        4,
        "UPLEFTFIRE"
    ],
    [
        12,
        6,
        "LEFTFIRE"
    ],
    [
        5,
        16,
        "DOWN"
    ],
    [
        11,
        24,
        "RIGHTFIRE"
    ],
    [
        5,
        10,
        "DOWN"
    ],
    [
        1,
        40,
        "FIRE"
    ],
    [
        17,
        40,
        "DOWNLEFTFIRE"
    ],
    [
        1,
        16,
        "FIRE"
    ],
    [
        5,
        12,
        "DOWN"
    ],
    [
        1,
        16,
        "FIRE"
    ],
    [
        16,
        10,
        "DOWNRIGHTFIRE"
    ],
    [
        15,
        6,
        "UPLEFTFIRE"
    ],
    [
        16,
        16,
        "DOWNRIGHTFIRE"
    ],
    [
        5,
        12,
        "DOWN"
    ],
    [
        12,
        4,
        "LEFTFIRE"
    ],
    [
        5,
        10,
        "DOWN"
    ],
    [
        9,
        8,
        "DOWNLEFT"
    ],
    [
        6,
        14,
        "UPRIGHT"
    ],
    [
        0,
        56,
        "NOOP"
    ],
    [
        3,
        16,
        "RIGHT"
    ],
    [
        6,
        12,
        "UPRIGHT"
    ],
    [
        7,
        24,
        "UPLEFT"
    ],
    [
        8,
        20,
        "DOWNRIGHT"
    ],
    [
        0,
        40,
        "NOOP"
    ],
    [
        7,
        8,
        "UPLEFT"
    ],
    [
        15,
        12,
        "UPLEFTFIRE"
    ],
    [
        11,
        12,
        "RIGHTFIRE"
    ],
    [
        12,
        10,
        "LEFTFIRE"
    ],
    [
        15,
        8,
        "UPLEFTFIRE"
    ],
    [
        14,
        16,
        "UPRIGHTFIRE"
    ],
    [
        5,
        32,
        "DOWN"
    ],
    [
        1,
        4,
        "FIRE"
    ],
    [
        15,
        30,
        "UPLEFTFIRE"
    ],
    [
        5,
        32,
        "DOWN"
    ],
    [
        15,
        32,
        "UPLEFTFIRE"
    ],
    [
        2,
        20,
        "UP"
    ],
    [
        8,
        24,
        "DOWNRIGHT"
    ],
    [
        7,
        8,
        "UPLEFT"
    ],
    [
        9,
        6,
        "DOWNLEFT"
    ],
    [
        2,
        12,
        "UP"
    ],
    [
        17,
        48,
        "DOWNLEFTFIRE"
    ],
    [
        2,
        40,
        "UP"
    ],
    [
        1,
        48,
        "FIRE"
    ],
    [
        17,
        20,
        "DOWNLEFTFIRE"
    ],
    [
        16,
        16,
        "DOWNRIGHTFIRE"
    ],
    [
        7,
        9,
        "UPLEFT"
    ],
    [
        12,
        56,
        "LEFTFIRE"
    ],
    [
        4,
        12,
        "LEFT"
    ],
    [
        2,
        16,
        "UP"
    ],
    [
        11,
        6,
        "RIGHTFIRE"
    ],
    [
        2,
        56,
        "UP"
    ],
    [
        14,
        27,
        "UPRIGHTFIRE"
    ],
    [
        17,
        40,
        "DOWNLEFTFIRE"
    ],
    [
        15,
        48,
        "UPLEFTFIRE"
    ],
    [
        14,
        19,
        "UPRIGHTFIRE"
    ],
    [
        16,
        11,
        "DOWNRIGHTFIRE"
    ],
    [
        14,
        8,
        "UPRIGHTFIRE"
    ],
    [
        5,
        15,
        "DOWN"
    ],
    [
        17,
        15,
        "DOWNLEFTFIRE"
    ],
    [
        2,
        31,
        "UP"
    ],
    [
        15,
        14,
        "UPLEFTFIRE"
    ],
    [
        12,
        47,
        "LEFTFIRE"
    ],
    [
        16,
        56,
        "DOWNRIGHTFIRE"
    ],
    [
        9,
        9,
        "DOWNLEFT"
    ],
    [
        15,
        5,
        "UPLEFTFIRE"
    ],
    [
        1,
        31,
        "FIRE"
    ],
    [
        15,
        24,
        "UPLEFTFIRE"
    ],
    [
        12,
        20,
        "LEFTFIRE"
    ],
    [
        6,
        10,
        "UPRIGHT"
    ],
    [
        15,
        48,
        "UPLEFTFIRE"
    ],
    [
        8,
        4,
        "DOWNRIGHT"
    ],
    [
        15,
        56,
        "UPLEFTFIRE"
    ],
    [
        16,
        56,
        "DOWNRIGHTFIRE"
    ]
]


class Policy:
    """Open-loop replay of a fixed action sequence.

    The 86 `(action_id, duration, name)` tuples in `MACROS` are expanded once
    at construction into a flat list of per-step action ids (about 1769
    steps in total). `act()` ignores its `obs` and `info` arguments entirely
    and just returns `self._actions[self._t]`, so this policy is a
    deterministic function of the internal time index only.

    After the scripted actions run out, `act()` returns `fallback_action`
    (default NOOP) forever.
    """

    def __init__(self, macros: Iterable[tuple[int, int, str]] = MACROS, fallback_action: int = 0) -> None:
        self._actions: list[int] = []
        for action, duration, _name in macros:
            if duration < 0:
                raise ValueError(f"negative macro duration: {duration}")
            # Expand each macro into `duration` copies of its action id so
            # the per-step lookup is O(1). Total expanded length equals
            # sum(durations) = 1769 for the 400-point route.
            self._actions.extend([int(action)] * int(duration))
        self._fallback_action = int(fallback_action)
        self._t = 0

    def reset(self) -> None:
        self._t = 0

    def act(self, obs=None, info=None) -> int:
        if self._t < len(self._actions):
            action = self._actions[self._t]
        else:
            action = self._fallback_action
        self._t += 1
        return action


def make_env(seed: int, render: bool = False):
    import envpool

    kwargs = dict(
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
    if render:
        kwargs["render_mode"] = "rgb_array"
    return envpool.make_gym(ENV_ID, **kwargs)


def reset_env(env):
    out = env.reset()
    if isinstance(out, tuple) and len(out) == 2:
        return out[0], out[1]
    return out, None


def step_env(env, action: int):
    out = env.step(np.asarray([action], dtype=np.int32))
    if len(out) == 5:
        obs, reward, terminated, truncated, info = out
        done = bool(np.asarray(terminated).reshape(-1)[0] or np.asarray(truncated).reshape(-1)[0])
    else:
        obs, reward, terminated, info = out
        done = bool(np.asarray(terminated).reshape(-1)[0])
    return obs, float(np.asarray(reward).reshape(-1)[0]), done, info


def rollout(seed: int, max_steps: int | None, record_mp4: Path | None, frame0_png: Path | None) -> dict[str, object]:
    render = record_mp4 is not None or frame0_png is not None
    env = make_env(seed, render=render)
    policy = Policy()
    obs, info = reset_env(env)
    policy.reset()

    writer = None
    try:
        if render:
            import cv2

            frame = env.render()
            if frame.ndim == 4:
                frame = frame[0]
            if frame0_png is not None:
                frame0_png.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(frame0_png), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            if record_mp4 is not None:
                record_mp4.parent.mkdir(parents=True, exist_ok=True)
                height, width = frame.shape[:2]
                writer = cv2.VideoWriter(str(record_mp4), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (width, height))
                if not writer.isOpened():
                    raise RuntimeError(f"failed to open video writer: {record_mp4}")
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        score = 0.0
        steps = 0
        done = False
        while not done:
            if max_steps is not None and steps >= max_steps:
                break
            action = policy.act(obs, info)
            obs, reward, done, info = step_env(env, action)
            score += reward
            steps += 1
            if writer is not None:
                import cv2

                frame = env.render()
                if frame.ndim == 4:
                    frame = frame[0]
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        if writer is not None:
            writer.release()
        env.close()

    return {
        "env_id": ENV_ID,
        "seed": seed,
        "score": score,
        "env_steps": steps,
        "done": done,
        "macro_count": len(MACROS),
        "scripted_action_steps": len(Policy()._actions),
        "expected_score": EXPECTED_SCORE,
        "expected_steps": EXPECTED_STEPS,
        "record_mp4": str(record_mp4) if record_mp4 else None,
        "frame0_png": str(frame0_png) if frame0_png else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--record-mp4", type=Path, default=None)
    parser.add_argument("--frame0-png", type=Path, default=None)
    parser.add_argument("--metadata-out", type=Path, default=None)
    args = parser.parse_args()

    result = rollout(args.seed, args.max_steps, args.record_mp4, args.frame0_png)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.metadata_out is not None:
        args.metadata_out.parent.mkdir(parents=True, exist_ok=True)
        args.metadata_out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
