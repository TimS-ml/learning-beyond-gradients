import argparse
import os
from collections.abc import Sequence
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np


ACT_NONE = 0
ACT_TURN_RIGHT = 1
ACT_TURN_LEFT = 2
ACT_FORWARD = 3
ACT_FORWARD_RIGHT = 4
ACT_FORWARD_LEFT = 5

PICKUP_HEALTH = 68.0
STAGE_AREA = 180.0


def detect_medikit(frame: np.ndarray) -> tuple[float, float, float, float, float] | None:
    gray = frame[..., 0]
    height, width = gray.shape
    mask = (gray > 150).astype(np.uint8) * 255
    mask[: int(height * 0.38), :] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 5), np.uint8))
    mask = cv2.dilate(mask, np.ones((2, 3), np.uint8), iterations=1)
    num, _, stats, cents = cv2.connectedComponentsWithStats(mask)
    candidates: list[tuple[float, float, float, float, float, float]] = []
    for i in range(1, num):
        x, y, w, h, area = stats[i]
        cx, cy = cents[i]
        aspect = w / max(1, h)
        if area < 10 or area > 3500:
            continue
        if y < height * 0.38 or h > height * 0.50 or w > width * 0.75:
            continue
        if aspect < 0.35 or aspect > 10:
            continue
        if w > width * 0.40 and h < height * 0.05:
            continue
        score = float(area) + 0.9 * float(cy)
        if 0.5 < aspect < 6.0:
            score += 30.0
        candidates.append((score, float(area), float(cx), float(cy), float(w), float(h)))
    if not candidates:
        return None
    return max(candidates)


def policy_action(step: int, frame: np.ndarray, health: float, state: dict[str, float]) -> int:
    height, width = frame.shape[:2]
    detected = detect_medikit(frame)
    if detected is None:
        if step - state.get("last_seen", -999.0) < 40:
            return ACT_TURN_LEFT if state.get("lost_spin", -1.0) < 0 else ACT_TURN_RIGHT
        return ACT_TURN_LEFT if (step // 35 + state["id"]) % 2 == 0 else ACT_TURN_RIGHT

    _, area, cx, cy, w, h = detected
    offset = cx - width / 2
    state["last_seen"] = float(step)
    state["lost_spin"] = -1.0 if cx < width / 2 else 1.0
    close = (
        area > STAGE_AREA
        or cy > height * 0.80
        or h > height * 0.12
        or w > width * 0.20
    )

    if close and health > PICKUP_HEALTH:
        if offset < -16:
            return ACT_TURN_LEFT
        if offset > 16:
            return ACT_TURN_RIGHT
        return ACT_NONE

    margin = 18 if close else 12
    if offset < -margin:
        return ACT_FORWARD_LEFT
    if offset > margin:
        return ACT_FORWARD_RIGHT
    return ACT_FORWARD


def make_grid(frames: np.ndarray, rewards: np.ndarray, info_rows: list[dict]) -> np.ndarray:
    frames = np.asarray(frames)
    n, h, w, c = frames.shape
    cols = 5
    rows = (n + cols - 1) // cols
    out = np.zeros((rows * h, cols * w, c), dtype=np.uint8)
    for i, frame in enumerate(frames):
        tile = frame.copy()
        info = info_rows[i] if i < len(info_rows) else {}
        health = float(info.get("HEALTH", 0.0))
        text = f"id {i} r {rewards[i]:.3f} hp {health:.0f}"
        cv2.putText(
            tile,
            text,
            (6, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        row, col = divmod(i, cols)
        out[row * h : (row + 1) * h, col * w : (col + 1) * w] = tile
    return out


def make_env(
    *,
    episodes: int,
    seed: int,
    render_width: int,
    render_height: int,
):
    import envpool.vizdoom as vizdoom_pkg
    import envpool.vizdoom.registration  # noqa: F401
    from envpool.registration import make_gym

    base = os.path.join(os.path.dirname(vizdoom_pkg.__file__), "maps")
    return make_gym(
        "D1Basic-v1",
        num_envs=episodes,
        seed=seed,
        wad_path=os.path.join(base, "D1_basic.wad"),
        cfg_path=os.path.join(base, "D1_basic.cfg"),
        use_combined_action=True,
        stack_num=1,
        frame_skip=1,
        max_episode_steps=2100,
        render_mode="rgb_array",
        render_width=render_width,
        render_height=render_height,
    )


def run(
    *,
    episodes: int,
    seed: int,
    record_mp4: str | None,
    render_width: int,
    render_height: int,
    fps: int,
) -> tuple[np.ndarray, np.ndarray]:
    env = make_env(
        episodes=episodes,
        seed=seed,
        render_width=render_width,
        render_height=render_height,
    )
    rewards = np.zeros(episodes)
    lengths = np.zeros(episodes)
    states = [
        {"id": float(i), "last_seen": -999.0, "lost_spin": -1.0}
        for i in range(episodes)
    ]
    final_info = [{} for _ in range(episodes)]
    ids: Sequence[int] = np.arange(episodes)
    writer = None
    try:
        _, active_info = env.reset()
        frames = env.render(env_ids=np.arange(episodes))
        if record_mp4:
            out = Path(record_mp4)
            out.parent.mkdir(parents=True, exist_ok=True)
            writer = imageio.get_writer(
                out, fps=fps, codec="libx264", quality=4, macro_block_size=1
            )
            writer.append_data(make_grid(frames, rewards, final_info))
        for step in range(2100):
            actions = []
            for row, env_id in enumerate(ids):
                health = float(np.asarray(active_info["HEALTH"])[row])
                actions.append(
                    policy_action(step, frames[int(env_id)], health, states[int(env_id)])
                )
            _, rew, terminated, truncated, info = env.step(
                np.asarray(actions, dtype=np.int64), ids
            )
            done = np.logical_or(terminated, truncated)
            cur_ids = np.asarray(info["env_id"])
            rewards[cur_ids] += rew
            lengths[cur_ids] += 1
            for row, env_id in enumerate(cur_ids):
                final_info[int(env_id)] = {
                    key: np.asarray(value)[row]
                    for key, value in info.items()
                    if np.asarray(value).ndim > 0 and len(np.asarray(value)) == len(done)
                }
            keep = ~done
            ids = cur_ids[keep]
            active_info = {
                key: np.asarray(value)[keep]
                for key, value in info.items()
                if np.asarray(value).ndim > 0 and len(np.asarray(value)) == len(done)
            }
            if len(ids) == 0:
                break
            frames = env.render(env_ids=np.arange(episodes))
            if writer is not None:
                writer.append_data(make_grid(frames, rewards, final_info))
    finally:
        if writer is not None:
            writer.close()
        env.close()
    return rewards, lengths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--record-mp4", default=None)
    parser.add_argument("--render-width", type=int, default=240)
    parser.add_argument("--render-height", type=int, default=180)
    parser.add_argument("--fps", type=int, default=35)
    args = parser.parse_args()

    rewards, lengths = run(
        episodes=args.episodes,
        seed=args.seed,
        record_mp4=args.record_mp4,
        render_width=args.render_width,
        render_height=args.render_height,
        fps=args.fps,
    )
    print("reward", rewards.tolist())
    print("mean", float(rewards.mean()), "min", float(rewards.min()))
    print("length", lengths.astype(int).tolist())


if __name__ == "__main__":
    main()
