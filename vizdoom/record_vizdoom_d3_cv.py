import importlib.util
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np


SPEC = importlib.util.spec_from_file_location(
    "d3cv", Path(__file__).with_name("heuristic_vizdoom_d3_cv.py")
)
d3cv = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(d3cv)


def grid(frames: np.ndarray, rewards: np.ndarray, info_rows: list[dict]) -> np.ndarray:
    frames = np.asarray(frames)
    n, h, w, c = frames.shape
    cols = 5
    rows = (n + cols - 1) // cols
    out = np.zeros((rows * h, cols * w, c), dtype=np.uint8)
    for i, frame in enumerate(frames):
        tile = frame.copy()
        info = info_rows[i] if i < len(info_rows) else {}
        text = (
            f"id {i} r {rewards[i]:.0f} "
            f"k {float(info.get('KILLCOUNT', 0)):.0f} "
            f"hp {float(info.get('HEALTH', 0)):.0f} "
            f"ammo {float(info.get('AMMO2', 0)):.0f}"
        )
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


def main() -> None:
    p = d3cv.BASE_P.copy()
    num_envs = 10
    out_path = Path("vizdoom/d3_cv_best_10seed_render_35fps.mp4")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cfg = d3cv.cfgfile(
        p.get("width", 640), p.get("height", 480), p.get("fmt", "CRCGCB")
    )
    env = d3cv.make_gym(
        "D3Battle-v1",
        num_envs=num_envs,
        seed=0,
        cfg_path=cfg,
        wad_path=d3cv.os.path.join(d3cv.BASE, "D3_battle.wad"),
        use_combined_action=False,
        stack_num=1,
        frame_skip=p["fs"],
        max_episode_steps=p["max_steps"],
        img_width=p.get("width", 640),
        img_height=p.get("height", 480),
        render_mode="rgb_array",
        render_width=240,
        render_height=180,
        reward_config={"DAMAGECOUNT": [1, 0], "KILLCOUNT": [10, 0]},
        selected_weapon_reward_config={},
        game_args=p.get("game_args", ""),
        force_speed=p.get("force_speed", False),
    )
    try:
        obs, active_info = env.reset()
        ids = np.arange(num_envs)
        rewards = np.zeros(num_envs)
        lengths = np.zeros(num_envs)
        states = [d3cv.State(p) for _ in range(num_envs)]
        final_info = [{} for _ in range(num_envs)]
        with imageio.get_writer(
            out_path, fps=35, codec="libx264", quality=5, macro_block_size=1
        ) as writer:
            writer.append_data(grid(env.render(env_ids=np.arange(num_envs)), rewards, final_info))
            for step in range(p["max_steps"]):
                actions = []
                for row, env_id in enumerate(ids):
                    row_info = {
                        key: np.asarray(value)[row]
                        for key, value in active_info.items()
                        if np.asarray(value).ndim > 0
                        and len(np.asarray(value)) == len(ids)
                    }
                    final_info[int(env_id)] = row_info
                    actions.append(
                        d3cv.choose(obs[row], row_info, states[int(env_id)], step, p)
                    )
                obs2, rew, term, trunc, info = env.step(np.asarray(actions), ids)
                done = np.logical_or(term, trunc)
                cur_ids = np.asarray(info["env_id"])
                rewards[cur_ids] += rew
                lengths[cur_ids] += 1
                frame = grid(env.render(env_ids=np.arange(num_envs)), rewards, final_info)
                for _ in range(int(p["fs"])):
                    writer.append_data(frame)
                keep = ~done
                ids = cur_ids[keep]
                obs = obs2[keep]
                active_info = {
                    key: np.asarray(value)[keep]
                    for key, value in info.items()
                    if np.asarray(value).ndim > 0
                    and len(np.asarray(value)) == len(done)
                }
                if len(ids) == 0:
                    break
    finally:
        env.close()
        d3cv.os.remove(cfg)
    print(out_path)
    print("reward", rewards.tolist())
    print("mean", float(rewards.mean()), "min", float(rewards.min()))
    print("length", lengths.astype(int).tolist())


if __name__ == "__main__":
    main()
