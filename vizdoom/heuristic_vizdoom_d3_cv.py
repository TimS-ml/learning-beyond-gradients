"""Solve VizDoom D3 Battle with a closed-loop CV+state-variable policy.

Reward is `1 * DAMAGECOUNT + 10 * KILLCOUNT`. The policy uses only:

- The rendered screen (via cv2 masks over three colour channels).
- The public EnvPool game variables `HEALTH`, `AMMO2`, `HITCOUNT`,
  `DAMAGECOUNT`, `KILLCOUNT`.

It does not read the WAD map, object coordinates, labels, or seed-specific
routes. Every behaviour (combat, kiting, item seeking, wall escape, stuck
detection, navigation) is a branch inside `choose()` that switches based on
the current CV detections, the running `State`, and the large `BASE_P`
config dict.

The 10-seed reward vector produced by this policy is
`[545, 475, 480, 440, 690, 500, 600, 595, 530, 715]` (`mean=557.0`,
`min=440.0`), matching the number quoted in the blog appendix.
"""

import os
import tempfile

import cv2
import envpool.vizdoom as v
import envpool.vizdoom.registration  # noqa: F401
import numpy as np
from envpool.registration import make_gym

# VizDoom's D3 Battle scenario ships as a fixed WAD + .cfg pair inside the
# EnvPool package. We only need to swap in a custom `available_buttons` block
# (below) so the policy can access the finer-grained turn/strafe primitives.
BASE = os.path.join(os.path.dirname(v.__file__), "maps")
ORIG_CFG = open(os.path.join(BASE, "D3_battle.cfg")).read()


def cfgfile(width=640, height=480, fmt="CRCGCB"):
    """Write a temporary VizDoom .cfg with our resolution/buttons overrides.

    Overrides applied:

    - `screen_resolution` -> requested `width x height`.
    - `screen_format` -> requested `fmt` (default CRCGCB, three per-channel
      chroma planes; the CV masks read these as int16).
    - `render_weapon = false`, `render_crosshair = false` -> keeps the
      screen bottom clear of HUD graphics that would confuse the CV.
    - `available_buttons` -> the finer combat/movement set the policy needs.
    """

    cfg = ORIG_CFG.replace("screen_resolution = RES_160X120", f"screen_resolution = RES_{width}X{height}")
    cfg = cfg.replace("screen_format = GRAY8", f"screen_format = {fmt}")
    cfg = cfg.replace("render_weapon = true", "render_weapon = false")
    cfg = cfg.replace("render_crosshair = true", "render_crosshair = false")
    custom = """available_buttons =
    {
        ATTACK
        SPEED
        MOVE_FORWARD
        MOVE_BACKWARD
        MOVE_RIGHT
        MOVE_LEFT
        TURN180
        TURN_LEFT_RIGHT_DELTA
    }

"""
    start = cfg.index("available_buttons")
    end = cfg.index("# Game variables")
    cfg = cfg[:start] + custom + cfg[end:]
    fd, path = tempfile.mkstemp(suffix=".cfg")
    os.close(fd)
    with open(path, "w") as f:
        f.write(cfg)
    return path


def image(obs):
    return obs.transpose(1, 2, 0)


def enemy_candidate(im, p):
    h, w = im.shape[:2]
    sx, sy, center = w / 320, h / 240, w / 2
    channels = [im[:, :, i].astype(np.int16) for i in range(3)]
    if p.get("enemy_mode", "crcgcb") == "rgb":
        c0, c1, c2 = channels
        strong = (
            (c0 > p["rgb_r"])
            & ((c0 - c1) > p["rgb_rg_margin"])
            & ((c0 - c2) > p["rgb_rb_margin"])
            & (c1 < p["rgb_g_max"])
            & (c2 < p["rgb_b_max"])
        )
    else:
        enemy_ch = channels[p["enemy_channel"]]
        other = [channels[i] for i in range(3) if i != p["enemy_channel"]]
        strong = (enemy_ch > p["c0"]) & ((enemy_ch - np.maximum(other[0], other[1])) > p["margin"]) & (other[0] < 55) & (other[1] < 55)
    if p.get("enemy_body_mode", False) and p["enemy_channel"] == 0:
        c0, c1, c2 = channels
        body = (
            (c0 > p["body_c0"])
            & ((c0 - c2) > p["body_c2_margin"])
            & (c1 < p["body_c1_max"])
            & (c2 < p["body_c2_max"])
        )
        mask = (strong | body).astype(np.uint8) * 255
    else:
        mask = strong.astype(np.uint8) * 255
    mask[: int(p.get("enemy_y1", 45) * sy)] = 0
    mask[int(p.get("enemy_y2", 205) * sy) :] = 0
    k = max(2, int(round(4 * sx)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    mask = cv2.dilate(mask, np.ones((max(2, int(4 * sy)), max(2, int(3 * sx))), np.uint8), iterations=1)
    n, _, stats, cent = cv2.connectedComponentsWithStats(mask, 8)
    best = None
    best_score = -1
    for j in range(1, n):
        _, _, bw, bh, area = stats[j]
        cx, cy = cent[j]
        if area < p["min_area"] * sx * sy or bh < 7 * sy or bw > 95 * sx or bh > 115 * sy:
            continue
        if (
            p.get("reject_pickup_enemy", False)
            and cy > p["pickup_enemy_y"] * sy
            and area < p["pickup_enemy_area"] * sx * sy
            and bh < p["pickup_enemy_h"] * sy
        ):
            continue
        if p.get("enemy_min_aspect", 0) and bh / max(float(bw), 1.0) < p["enemy_min_aspect"]:
            continue
        score = area * (bh / (16 * sy)) ** p["enemy_h_exp"]
        score *= 1 + p["enemy_y_weight"] * (cy - p["enemy_y_ref"] * sy) / (180 * sy)
        score /= 1 + abs(cx - center) / (p["enemy_center_bias"] * sx)
        if score > best_score:
            best_score = score
            best = (cx, cy, bw, bh, area, score)
    return best


def item_candidate(im, p, ammo=None, hp=None):
    h, w = im.shape[:2]
    sx, sy, center = w / 320, h / 240, w / 2
    c0 = im[:, :, 0].astype(np.int16)
    c1 = im[:, :, 1].astype(np.int16)
    c2 = im[:, :, 2].astype(np.int16)
    channels = [c0, c1, c2]
    enemy_ch = channels[p["enemy_channel"]]
    other = [channels[i] for i in range(3) if i != p["enemy_channel"]]
    enemy = (enemy_ch > p["c0"]) & ((enemy_ch - np.maximum(other[0], other[1])) > p["margin"]) & (other[0] < 55) & (other[1] < 55)
    if p.get("item_mode", "diff") == "bright":
        item_b0 = p["item_b0"]
        if (
            ammo is not None
            and hp is not None
            and (ammo <= p.get("item_loose_ammo", -1) or hp <= p.get("item_loose_hp", -1))
        ):
            item_b0 = p.get("item_loose_b0", item_b0)
        medikit = (c0 > item_b0) & (c1 > p["item_b1"]) & (c2 > p["item_b2"])
        clip = (c0 > p["clip_b0"]) & (c1 > p["clip_b1"]) & ((c0 - c2) > p["clip_delta"])
        mask = ((medikit | clip) & (~enemy)).astype(np.uint8) * 255
        mask[: int(50 * sy)] = 0
        mask[int(218 * sy) :] = 0
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((max(2, int(3 * sy)), max(2, int(3 * sx))), np.uint8))
        n, _, stats, cent = cv2.connectedComponentsWithStats(mask, 8)
        best = None
        best_score = -1
        for j in range(1, n):
            _, _, bw, bh, area = stats[j]
            cx, cy = cent[j]
            if area < p["item_min_area"] * sx * sy or area > p["item_max_area"] * sx * sy:
                continue
            if bw > p["item_max_w"] * sx or bh > p["item_max_h"] * sy or bw < 2 * sx or bh < 2 * sy:
                continue
            score = area * (1 + (cy - 85 * sy) / (130 * sy))
            score /= 1 + abs(cx - center) / (85 * sx)
            if score > best_score:
                best_score = score
                best = (cx, cy, bw, bh, area, score)
        if best is not None:
            return best
        if not p.get("item_bright_fallback", True):
            return None

    diff = np.maximum.reduce([abs(c0 - c1), abs(c0 - c2), abs(c1 - c2)])
    mask = ((diff > p["item_diff"]) & (~enemy)).astype(np.uint8) * 255
    mask[: int(55 * sy)] = 0
    mask[int(205 * sy) :] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((max(2, int(3 * sy)), max(2, int(3 * sx))), np.uint8))
    n, _, stats, cent = cv2.connectedComponentsWithStats(mask, 8)
    best = None
    best_score = -1
    for j in range(1, n):
        _, _, bw, bh, area = stats[j]
        cx, cy = cent[j]
        if area < 8 * sx * sy or area > 700 * sx * sy or bw > 52 * sx or bh > 42 * sy:
            continue
        score = area * (1 + (cy - 80 * sy) / (140 * sy))
        score /= 1 + abs(cx - center) / (90 * sx)
        if score > best_score:
            best_score = score
            best = (cx, cy, bw, bh, area, score)
    return best


def threat_candidate(im, p):
    h, w = im.shape[:2]
    sx, sy, center = w / 320, h / 240, w / 2
    c0 = im[:, :, 0].astype(np.int16)
    c1 = im[:, :, 1].astype(np.int16)
    c2 = im[:, :, 2].astype(np.int16)
    mask = ((c0 > p["threat_b0"]) & (c1 > p["threat_b1"]) & (c2 > p["threat_b2"])).astype(np.uint8) * 255
    mask[: int(45 * sy)] = 0
    mask[int(230 * sy) :] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((max(2, int(4 * sy)), max(2, int(4 * sx))), np.uint8))
    n, _, stats, cent = cv2.connectedComponentsWithStats(mask, 8)
    best = None
    best_score = -1
    for j in range(1, n):
        _, _, bw, bh, area = stats[j]
        cx, cy = cent[j]
        if area < p["threat_area"] * sx * sy or bw < p["threat_w"] * sx or bh < p["threat_h"] * sy:
            continue
        if bw > p["threat_max_w"] * sx or bh > p["threat_max_h"] * sy:
            continue
        score = area / (1 + abs(cx - center) / (110 * sx))
        if score > best_score:
            best_score = score
            best = (cx, cy, bw, bh, area, score)
    return best


def projectile_candidate(im, p):
    h, w = im.shape[:2]
    sx, sy, center = w / 320, h / 240, w / 2
    c0 = im[:, :, 0].astype(np.int16)
    c1 = im[:, :, 1].astype(np.int16)
    c2 = im[:, :, 2].astype(np.int16)
    mask = ((c0 > p["proj_b0"]) & (c1 > p["proj_b1"]) & (c2 > p["proj_b2"])).astype(np.uint8) * 255
    mask[: int(35 * sy)] = 0
    mask[int(220 * sy) :] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((max(2, int(3 * sy)), max(2, int(3 * sx))), np.uint8))
    n, _, stats, cent = cv2.connectedComponentsWithStats(mask, 8)
    best = None
    best_score = -1
    for j in range(1, n):
        _, _, bw, bh, area = stats[j]
        cx, cy = cent[j]
        if area < p["proj_min_area"] * sx * sy or area > p["proj_max_area"] * sx * sy:
            continue
        if bw > p["proj_max_w"] * sx or bh > p["proj_max_h"] * sy:
            continue
        score = area * (1 + (cy - 60 * sy) / (160 * sy))
        score /= 1 + abs(cx - center) / (120 * sx)
        if score > best_score:
            best_score = score
            best = (cx, cy, bw, bh, area, score)
    return best


def nav_candidate(im, p):
    h, w = im.shape[:2]
    sx, sy, center = w / 320, h / 240, w / 2
    gray = im.astype(np.int16).mean(axis=2)
    y1 = int(p["nav_y1"] * sy)
    y2 = int(p["nav_y2"] * sy)
    band = gray[y1:y2]
    open_mask = (band < p["nav_dark"]).astype(np.uint8) * 255
    open_mask = cv2.morphologyEx(open_mask, cv2.MORPH_OPEN, np.ones((max(2, int(3 * sy)), max(2, int(3 * sx))), np.uint8))
    open_mask = cv2.morphologyEx(open_mask, cv2.MORPH_CLOSE, np.ones((max(2, int(7 * sy)), max(2, int(5 * sx))), np.uint8))
    n, _, stats, cent = cv2.connectedComponentsWithStats(open_mask, 8)
    best = None
    best_score = -1
    for j in range(1, n):
        x, y, bw, bh, area = stats[j]
        cx, cy = cent[j]
        if area < p["nav_min_area"] * sx * sy or bh < p["nav_min_h"] * sy or bw < p["nav_min_w"] * sx:
            continue
        # Prefer near-horizon openings but keep a mild center prior.
        score = area * (1 + (y + bh) / max(1, y2 - y1))
        score /= 1 + abs(cx - center) / (p["nav_center_bias"] * sx)
        if score > best_score:
            best_score = score
            best = (cx, y1 + cy, bw, bh, area, score)
    return best


def wall_avoid_turn(im, p):
    h, w = im.shape[:2]
    sx, sy = w / 320, h / 240
    gray = im.astype(np.int16).mean(axis=2)
    y1 = int(p["wall_y1"] * sy)
    y2 = int(p["wall_y2"] * sy)
    cx = w // 2
    cw = int(p["wall_center_w"] * sx)
    sw = int(p["wall_side_w"] * sx)
    center = gray[y1:y2, max(0, cx - cw) : min(w, cx + cw)]
    left = gray[y1:y2, max(0, cx - cw - sw) : max(0, cx - cw)]
    right = gray[y1:y2, min(w, cx + cw) : min(w, cx + cw + sw)]
    if center.size == 0 or left.size == 0 or right.size == 0:
        return None
    center_dark = float(np.mean(center < p["wall_dark"]))
    left_dark = float(np.mean(left < p["wall_dark"]))
    right_dark = float(np.mean(right < p["wall_dark"]))
    # Low dark/open fraction in the center means we are looking at a nearby wall.
    if center_dark > p["wall_blocked_dark_max"]:
        return None
    delta = right_dark - left_dark
    if abs(delta) < p["wall_side_delta"]:
        return 0.0
    return np.sign(delta) * p["wall_turn"]


def close_wall_turn(im, p):
    h, w = im.shape[:2]
    sx, sy = w / 320, h / 240
    gray = im.astype(np.int16).mean(axis=2)
    y1 = int(p["escape_y1"] * sy)
    y2 = int(p["escape_y2"] * sy)
    cx = w // 2
    cw = int(p["escape_center_w"] * sx)
    sw = int(p["escape_side_w"] * sx)
    center = gray[y1:y2, max(0, cx - cw) : min(w, cx + cw)]
    left = gray[y1:y2, max(0, cx - cw - sw) : max(0, cx - cw)]
    right = gray[y1:y2, min(w, cx + cw) : min(w, cx + cw + sw)]
    if center.size == 0 or left.size == 0 or right.size == 0:
        return None
    center_mean = float(center.mean())
    center_dark = float(np.mean(center < p["escape_dark"]))
    edge = float(np.mean(np.abs(np.diff(center, axis=1)) > p["escape_edge_delta"]))
    if (
        center_mean < p["escape_mean_min"]
        or center_dark > p["escape_center_dark_max"]
        or edge > p["escape_edge_max"]
    ):
        return None
    left_open = float(np.mean(left < p["escape_side_dark"]))
    right_open = float(np.mean(right < p["escape_side_dark"]))
    if max(left_open, right_open) < p["escape_side_open_min"]:
        return 0.0
    delta = right_open - left_open
    if abs(delta) < p["escape_side_delta"]:
        return 0.0
    return np.sign(delta) * p["escape_turn"]


class State:
    """Per-env recurrent state carried across every `choose()` call.

    Grouped by concern:

    - Combat: `damage`, `hit`, `health` (previous values from info, used to
      detect deltas), `lock` (aiming lock countdown), `bad` (number of bad
      shots taken), `dx` (last horizontal offset to an enemy), `panic`.
    - Movement / anti-stuck: `turn` (current turn direction sign), `prev`
      (previous frame, for bump/stuck detection), `stuck`, `bump`, `bounce`
      / `bounce_i` / `bounce_turn` (random-bounce escape state).
    - Navigation: `arc_turn`, `bored_after`, `bored_turn` (adapted from
      defaults by the `adapt_arc` branch), `close_area`, `close_h`,
      `nav_i` / `nav_until` / `nav_turn` (telegraph nav state), and
      `wall_escape` / `wall_escape_turn` (close-wall escape state).
    - Novelty: `recent_views` (rolling list of hashed low-res frames used to
      detect that the agent has visited the same visual state recently).
    - Open-space commitment: `open_commit` / `open_turn` (frames left to
      commit to an "open hallway" direction before re-planning).
    - Progress tracking: `last_progress` (last step index where reward
      changed; every "bored" branch is gated on how long ago this was).
    """

    def __init__(self, p=None):
        self.damage = 0
        self.hit = 0
        self.health = 100
        self.lock = 0
        self.bad = 0
        self.dx = 0
        self.turn = 1 if p is None else p.get("init_turn", 1)
        self.panic = 0
        self.prev = None
        self.stuck = 0
        self.last_progress = 0
        self.adapted = False
        self.arc_turn = None
        self.bored_after = None
        self.bored_turn = None
        self.close_adapted = False
        self.close_area = None
        self.close_h = None
        self.bump = 0
        self.bounce = 0
        self.bounce_i = 0
        self.bounce_turn = 0.0
        self.nav_i = 0
        self.nav_until = 0
        self.nav_turn = self.turn * (1.0 if p is None else p.get("telegraph_turn_init", p.get("arc_turn", 1.0)))
        self.recent_views = []
        self.wall_escape = 0
        self.wall_escape_turn = 0.0
        self.open_commit = 0
        self.open_turn = 0.0


def action(attack=0, speed=0, fw=0, back=0, right=0, left=0, turn=0.0, turn180=0):
    """Build an EnvPool D3 combined-action vector in `available_buttons` order.

    Argument order maps to the buttons defined in `cfgfile()`:
    ATTACK, SPEED, MOVE_FORWARD, MOVE_BACKWARD, MOVE_RIGHT, MOVE_LEFT,
    TURN180, TURN_LEFT_RIGHT_DELTA. `turn` is a signed continuous value; the
    other flags are 0/1 button presses.
    """

    return np.asarray([attack, speed, fw, back, right, left, turn180, turn], dtype=np.float64)


def strafe_side(step, p):
    mode = p.get("strafe_mode", "alternate")
    if mode == "left":
        return "left"
    if mode == "right":
        return "right"
    period = max(1, int(p.get("strafe_period", 8)))
    return "right" if (step // period) % 2 else "left"


def side_action_kwargs(side):
    if side == "left":
        return {"left": 1}
    if side == "right":
        return {"right": 1}
    return {}


def nav_side(step, st, p):
    mode = p.get("nav_strafe", "none")
    if mode == "none":
        return "none"
    if mode == "turn":
        return "left" if st.turn < 0 else "right"
    if mode == "opposite_turn":
        return "right" if st.turn < 0 else "left"
    if mode == "alternate":
        period = max(1, int(p.get("nav_strafe_period", 24)))
        return "right" if (step // period) % 2 else "left"
    return mode


def start_bounce(st, p):
    st.bounce_i += 1
    frac = (st.bounce_i * 0.6180339887498949) % 1.0
    mag = p["bounce_turn_min"] + frac * (p["bounce_turn_max"] - p["bounce_turn_min"])
    st.turn = -st.turn
    st.bounce_turn = st.turn * mag
    st.bounce = p["bounce_steps"]
    st.bump = 0


def bounce_nav(im, st, step, p):
    if st.prev is not None:
        diff = float(np.mean(np.abs(im.astype(np.int16) - st.prev.astype(np.int16))))
        st.bump = st.bump + 1 if diff < p["bounce_diff"] else max(0, st.bump - 1)
    st.prev = im.copy()

    if st.bounce > 0:
        st.bounce -= 1
        if st.bounce > p["bounce_back_steps"]:
            return action(speed=1, back=1, turn=st.bounce_turn)
        return action(speed=1, fw=1, turn=st.bounce_turn * p["bounce_exit_turn_scale"])

    if st.bump >= p["bounce_bump_limit"]:
        start_bounce(st, p)
        return action(speed=1, back=1, turn=st.bounce_turn, turn180=1 if p.get("bounce_turn180", False) else 0)

    if p.get("bounce_micro_jitter", 0):
        frac = ((step + 1) * 0.3819660112501051) % 1.0
        jitter = (frac - 0.5) * p["bounce_micro_jitter"]
    else:
        jitter = 0.0
    return action(speed=1, fw=1, turn=st.turn * p["bounce_arc_turn"] + jitter)


def telegraph_nav(im, st, step, p):
    if st.prev is not None:
        diff = float(np.mean(np.abs(im.astype(np.int16) - st.prev.astype(np.int16))))
        st.bump = st.bump + 1 if diff < p["telegraph_bump_diff"] else max(0, st.bump - 1)
    st.prev = im.copy()

    if st.bounce > 0:
        st.bounce -= 1
        if st.bounce > p["bounce_back_steps"]:
            return action(speed=1, back=1, turn=st.bounce_turn)
        return action(speed=1, fw=1, turn=st.bounce_turn * p["bounce_exit_turn_scale"])

    if st.bump >= p["telegraph_bump_limit"]:
        start_bounce(st, p)
        return action(speed=1, back=1, turn=st.bounce_turn, turn180=1 if p.get("bounce_turn180", False) else 0)

    if step >= st.nav_until:
        st.nav_i += 1
        frac = (st.nav_i * 0.6180339887498949) % 1.0
        frac2 = (st.nav_i * 0.4142135623730951) % 1.0
        frac3 = (st.nav_i * 0.7548776662466927) % 1.0
        duration = p["telegraph_min_steps"] + int(frac * (p["telegraph_max_steps"] - p["telegraph_min_steps"]))
        mag = p["telegraph_turn_min"] + frac2 * (p["telegraph_turn_max"] - p["telegraph_turn_min"])
        sign = -1.0 if frac3 < 0.5 else 1.0
        st.nav_turn = sign * mag
        st.nav_until = step + duration

    return action(speed=1, fw=1, turn=st.nav_turn)


def view_hash(im, p):
    gray = im.astype(np.uint8).mean(axis=2).astype(np.uint8)
    small = cv2.resize(gray, (p["novelty_w"], p["novelty_h"]), interpolation=cv2.INTER_AREA)
    return (small > small.mean()).astype(np.uint8).tobytes()


def choose(obs, info, st, step, p):
    im = image(obs)
    h, w = im.shape[:2]
    sx, sy, center = w / 320, h / 240, w / 2
    ammo = float(info.get("AMMO2", 20))
    hp = float(info.get("HEALTH", 100))
    dmg = float(info.get("DAMAGECOUNT", 0))
    hit = float(info.get("HITCOUNT", 0))
    if dmg > st.damage or hit > st.hit:
        st.lock = p["hit_lock"]
        st.bad = 0
        st.last_progress = step
    elif st.lock > 0:
        st.bad += 1
    if hp < st.health - 0.1:
        st.panic = p["panic"]
        st.turn *= -1
    st.damage, st.hit, st.health = dmg, hit, hp

    enemy = enemy_candidate(im, p)
    if p.get("urgent_item_before_enemy", False) and (ammo <= p["urgent_ammo"] or hp <= p["urgent_hp"]):
        urgent_item = item_candidate(im, p, ammo, hp)
        if urgent_item is not None:
            take_item = enemy is None or ammo <= p["urgent_item_ignore_enemy_ammo"]
            if enemy is not None and not take_item:
                ecx, _, _, ebh, earea, _ = enemy
                edx = abs((ecx - center) / sx)
                take_item = (
                    edx >= p["urgent_enemy_dx"]
                    or (earea < p["urgent_enemy_area"] * sx * sy and ebh < p["urgent_enemy_h"] * sy)
                )
            if take_item:
                cx, _, _, _, _, _ = urgent_item
                dx = (cx - center) / sx
                turn = float(np.clip(dx * p["item_k"], -p["max_turn"], p["max_turn"]))
                return action(speed=1, fw=1, turn=turn)

    if p.get("avoid_threat", False) and hp <= p["avoid_hp"]:
        threat = threat_candidate(im, p)
        if threat is not None:
            cx, _, _, _, _, _ = threat
            dx = (cx - center) / sx
            turn = float(np.clip(-dx * p["threat_turn_k"], -p["max_turn"], p["max_turn"]))
            if dx < 0:
                return action(speed=1, back=1, right=1, turn=turn)
            return action(speed=1, back=1, left=1, turn=turn)

    if enemy is not None and ammo > 0:
        cx, _, _, bh, area, _ = enemy
        dx = (cx - center) / sx
        st.dx = dx
        st.lock = max(st.lock, p["seen_lock"])
        max_turn = p["far_max_turn"] if abs(dx) >= p["far_turn_dx"] else p["max_turn"]
        turn = float(np.clip(dx * p["aim_k"], -max_turn, max_turn))
        close_area = p["close_area"] if st.close_area is None else st.close_area
        close_h = p["close_h"] if st.close_h is None else st.close_h
        if abs(dx) > p["shoot_tol"]:
            track_attack = (
                p.get("track_attack", False)
                and ammo >= p["track_attack_ammo"]
                and abs(dx) <= p["track_attack_dx"]
                and area >= p["track_attack_area"] * sx * sy
            )
            # Track without spending bullets unless almost lined up.
            danger = (
                area > p["track_close_area"] * sx * sy
                or bh > p["track_close_h"] * sy
                or (hp <= p["track_hp"] and ammo >= p.get("track_hp_min_ammo", 0))
                or (p.get("track_panic", False) and st.panic > 0)
            )
            if danger or p.get("track_mode", "forward") == "back":
                if dx < 0:
                    return action(attack=1 if track_attack else 0, speed=1, back=1, right=1, turn=turn)
                return action(attack=1 if track_attack else 0, speed=1, back=1, left=1, turn=turn)
            if p.get("track_mode", "forward") == "strafe":
                if dx < 0:
                    return action(attack=1 if track_attack else 0, speed=1, right=1, turn=turn)
                return action(attack=1 if track_attack else 0, speed=1, left=1, turn=turn)
            if p.get("track_mode", "forward") == "stop":
                return action(attack=1 if track_attack else 0, speed=1, turn=turn)
            return action(attack=1 if track_attack else 0, speed=1, fw=1, turn=turn)
        if p["kite"] or area > close_area * sx * sy or bh > close_h * sy or st.panic > 0:
            st.panic = max(0, st.panic - 1)
            if strafe_side(step, p) == "right":
                return action(attack=1, speed=1, back=1, right=1, turn=turn)
            return action(attack=1, speed=1, back=1, left=1, turn=turn)
        if (
            p.get("center_move", "strafe") == "forward_if_far"
            and area < p["center_fw_area"] * sx * sy
            and bh < p["center_fw_h"] * sy
        ):
            return action(attack=1, speed=1, fw=1, turn=turn)
        if p.get("center_move", "strafe") == "forward":
            return action(attack=1, speed=1, fw=1, turn=turn)
        if p.get("center_move", "strafe") == "forward_strafe":
            if strafe_side(step, p) == "right":
                return action(attack=1, speed=1, fw=1, right=1, turn=turn)
            return action(attack=1, speed=1, fw=1, left=1, turn=turn)
        if strafe_side(step, p) == "right":
            return action(attack=1, speed=1, right=1, turn=turn)
        return action(attack=1, speed=1, left=1, turn=turn)

    if st.lock > 0 and ammo > 0:
        st.lock -= 1
        turn = float(np.clip(st.dx * p["aim_k"], -p["max_turn"], p["max_turn"]))
        if abs(st.dx) <= p["shoot_tol"] and st.bad < p["bad_limit"]:
            return action(attack=1, turn=turn)
        if st.bad >= p["bad_limit"]:
            st.turn *= -1
            st.bad = 0
        return action(speed=1, turn=turn if abs(st.dx) > p["shoot_tol"] else st.turn * p["scan_turn"])

    if p.get("projectile_target", False) and ammo > 0:
        proj = projectile_candidate(im, p)
        if proj is not None:
            cx, _, _, _, _, _ = proj
            dx = (cx - center) / sx
            turn = float(np.clip(dx * p["proj_turn_k"], -p["max_turn"], p["max_turn"]))
            if abs(dx) <= p["proj_shoot_tol"]:
                side = strafe_side(step, p)
                return action(attack=1, speed=1, left=1 if side == "left" else 0, right=1 if side == "right" else 0, turn=turn)
            return action(speed=1, fw=1, turn=turn)

    if p.get("projectile_scan", False) and ammo > 0:
        proj = projectile_candidate(im, p)
        if proj is not None:
            cx, _, _, _, _, _ = proj
            dx = (cx - center) / sx
            turn = float(np.clip(dx * p["proj_turn_k"], -p["max_turn"], p["max_turn"]))
            if p.get("projectile_scan_dodge", False):
                if dx < 0:
                    return action(speed=1, back=1 if hp <= p["projectile_scan_back_hp"] else 0, right=1, turn=turn)
                return action(speed=1, back=1 if hp <= p["projectile_scan_back_hp"] else 0, left=1, turn=turn)
            return action(speed=1, turn=turn)

    if st.panic > 0:
        st.panic -= 1
        return action(speed=1, back=1, turn=st.turn * p["scan_turn"])

    if p.get("close_wall_escape", False) and step - st.last_progress > p["escape_after_progress"]:
        if st.wall_escape > 0:
            st.wall_escape -= 1
            turn = st.wall_escape_turn
            if st.wall_escape > p["escape_forward_steps"]:
                return action(speed=1, back=1, turn=turn, turn180=1 if p.get("escape_turn180", False) and st.wall_escape == p["escape_steps"] - 1 else 0)
            return action(speed=1, fw=1, turn=turn * p["escape_exit_turn_scale"])
        wall_turn = close_wall_turn(im, p)
        if wall_turn is not None:
            if wall_turn == 0.0:
                st.turn *= -1
                wall_turn = st.turn * p["escape_turn"]
            else:
                st.turn = 1 if wall_turn > 0 else -1
            st.wall_escape_turn = wall_turn
            st.wall_escape = p["escape_steps"]
            return action(speed=1, back=1, turn=wall_turn, turn180=1 if p.get("escape_turn180", False) else 0)

    checked_stuck = False
    if p.get("stuck_before_item", False):
        checked_stuck = True
        if st.prev is not None:
            diff = float(np.mean(np.abs(im.astype(np.int16) - st.prev.astype(np.int16))))
            st.stuck = st.stuck + 1 if diff < p["stuck_diff"] else max(0, st.stuck - 1)
        st.prev = im.copy()
        if st.stuck > p["stuck_limit"]:
            st.stuck = 0
            if p.get("stuck_random_bounce", False):
                start_bounce(st, p)
                return action(speed=1, back=1, turn=st.bounce_turn, turn180=1 if p.get("bounce_turn180", False) else 0)
            st.turn *= -1
            return action(speed=1, back=1, turn=st.turn * p["scan_turn"], turn180=1 if p.get("turn180_stuck", False) else 0)

    item = item_candidate(im, p, ammo, hp) if (ammo <= p["ammo_seek"] or hp <= p["hp_seek"]) else None
    if item is not None:
        cx, _, _, _, _, _ = item
        dx = (cx - center) / sx
        turn = float(np.clip(dx * p["item_k"], -p["max_turn"], p["max_turn"]))
        return action(speed=1, fw=1, turn=turn)

    if p.get("open_when_bored", False) and step - st.last_progress > p["open_after"]:
        if p.get("open_commit_steps", 0) and st.open_commit > 0:
            st.open_commit -= 1
            return action(speed=1, fw=1, turn=st.open_turn)
        nav = nav_candidate(im, p)
        if nav is not None:
            cx, _, _, _, _, _ = nav
            dx = (cx - center) / sx
            turn = float(np.clip(dx * p["open_k"], -p["open_max_turn"], p["open_max_turn"]))
            if p.get("open_commit_steps", 0):
                st.open_commit = p["open_commit_steps"]
                st.open_turn = turn
            if abs(dx) <= p["open_center_tol"]:
                return action(speed=1, fw=1)
            return action(speed=1, fw=1, turn=turn)

    if p.get("stuck_random_bounce", False) and st.bounce > 0:
        st.bounce -= 1
        if st.bounce > p["bounce_back_steps"]:
            return action(speed=1, back=1, turn=st.bounce_turn)
        return action(speed=1, fw=1, turn=st.bounce_turn * p["bounce_exit_turn_scale"])

    if p.get("nav_mode", "arc") == "bounce":
        return bounce_nav(im, st, step, p)

    if p.get("nav_mode", "arc") == "telegraph":
        return telegraph_nav(im, st, step, p)

    if p.get("novelty_bounce", False):
        vh = view_hash(im, p)
        if p.get("novelty_hamming", 0) <= 0:
            repeated = vh in st.recent_views
        else:
            cur = np.frombuffer(vh, dtype=np.uint8)
            repeated = any(
                int(np.count_nonzero(cur != np.frombuffer(old, dtype=np.uint8))) <= p["novelty_hamming"]
                for old in st.recent_views
            )
        st.recent_views.append(vh)
        if len(st.recent_views) > p["novelty_memory"]:
            st.recent_views.pop(0)
        if repeated and step - st.last_progress > p["novelty_after"]:
            start_bounce(st, p)
            return action(speed=1, back=1, turn=st.bounce_turn, turn180=1 if p.get("bounce_turn180", False) else 0)

    if not checked_stuck and st.prev is not None:
        diff = float(np.mean(np.abs(im.astype(np.int16) - st.prev.astype(np.int16))))
        st.stuck = st.stuck + 1 if diff < p["stuck_diff"] else max(0, st.stuck - 1)
    if not checked_stuck:
        st.prev = im.copy()
        if st.stuck > p["stuck_limit"]:
            st.stuck = 0
            if p.get("stuck_random_bounce", False):
                start_bounce(st, p)
                return action(speed=1, back=1, turn=st.bounce_turn, turn180=1 if p.get("bounce_turn180", False) else 0)
            st.turn *= -1
            return action(speed=1, back=1, turn=st.turn * p["scan_turn"], turn180=1 if p.get("turn180_stuck", False) else 0)

    if p.get("adapt_arc", False) and (not st.adapted) and step >= p["adapt_step"]:
        high_progress = (
            p.get("adapt_fast", False)
            and hp >= p["adapt_fast_hp"]
            and dmg >= p["adapt_fast_damage"]
        )
        low_progress = dmg <= p["adapt_low_damage"] and hp >= p["adapt_high_hp"]
        danger_progress = hp <= p["adapt_low_hp"] and dmg >= p["adapt_min_damage"]
        if high_progress:
            st.arc_turn = p["adapt_fast_arc_turn"]
            st.bored_after = p["adapt_fast_bored_after"]
            st.bored_turn = p["adapt_fast_bored_turn"]
        elif low_progress or danger_progress:
            if p.get("adapt_flip_turn", False):
                st.turn *= -1
            st.arc_turn = p["adapt_arc_turn"]
            st.bored_after = p["adapt_bored_after"]
            st.bored_turn = p["adapt_bored_turn"]
        if p.get("adapt_once", False) or high_progress or low_progress or danger_progress:
            st.adapted = True

    if p.get("adapt_close", False) and (not st.close_adapted) and step >= p["adapt_close_step"]:
        good_pressure = (
            hp >= p["adapt_close_hp"]
            and ammo >= p["adapt_close_ammo"]
            and p["adapt_close_damage_min"] <= dmg <= p["adapt_close_damage_max"]
        )
        if good_pressure:
            st.close_area = p["adapt_close_area"]
            st.close_h = p["adapt_close_h"]
        if p.get("adapt_close_once", True) or good_pressure:
            st.close_adapted = True

    arc_turn = p["arc_turn"] if st.arc_turn is None else st.arc_turn
    bored_after = p["bored_after"] if st.bored_after is None else st.bored_after
    bored_turn = p["bored_turn"] if st.bored_turn is None else st.bored_turn
    bored_arc_turn = p.get("bored_arc_turn", arc_turn)
    if (
        p.get("bored_context_arc", False)
        and dmg >= p["bored_context_damage"]
        and ammo >= p["bored_context_ammo"]
        and hp >= p["bored_context_hp"]
    ):
        bored_arc_turn = p["bored_context_arc_turn"]

    if bored_after and step - st.last_progress > bored_after:
        bored_phase = (step - st.last_progress - bored_after) % (bored_turn + p["bored_forward"])
        if bored_phase < bored_turn:
            if p.get("turn180_bored", False) and bored_phase == 0:
                st.turn *= -1
                return action(speed=1, turn180=1)
            wall_turn = wall_avoid_turn(im, p) if p.get("wall_bored", False) else None
            if wall_turn is not None:
                if wall_turn == 0.0:
                    wall_turn = st.turn * p["bored_scan_turn"]
                return action(speed=1, turn=wall_turn)
            return action(speed=1, turn=st.turn * p["bored_scan_turn"])
        side = side_action_kwargs(nav_side(step, st, p))
        return action(speed=1, fw=1, turn=st.turn * bored_arc_turn, **side)

    if p.get("nav_mode", "arc") == "open":
        nav = nav_candidate(im, p)
        if nav is not None:
            cx, _, bw, _, area, _ = nav
            dx = (cx - center) / sx
            turn = float(np.clip(dx * p["nav_k"], -p["nav_max_turn"], p["nav_max_turn"]))
            if abs(dx) <= p["nav_center_tol"]:
                return action(speed=1, fw=1)
            return action(speed=1, fw=1, turn=turn)

    if p.get("wall_avoid", False):
        turn = wall_avoid_turn(im, p)
        if turn is not None:
            if turn == 0.0:
                st.turn *= -1
                turn = st.turn * p["wall_turn"]
            return action(speed=1, back=1 if p.get("wall_back", False) else 0, turn=turn)

    phase = step % p["period"]
    if p["mode"] == "turret":
        if p["noise_period"] and ammo >= p["noise_ammo"] and step % p["noise_period"] == 0:
            return action(attack=1, speed=1, turn=st.turn * p["scan_turn"])
        return action(speed=1, turn=st.turn * p["scan_turn"])
    if phase < p["forward"]:
        if p["noise_period"] and ammo >= p["noise_ammo"] and step % p["noise_period"] == 0:
            return action(attack=1, speed=1, fw=1)
        arc = st.turn * arc_turn if p["arc_mod"] and phase % p["arc_mod"] == 0 else 0.0
        side = side_action_kwargs(nav_side(step, st, p))
        return action(speed=1, fw=1, turn=arc, **side)
    if phase < p["forward"] + p["turn"]:
        return action(speed=1, turn=st.turn * p["scan_turn"])
    if phase == p["forward"] + p["turn"]:
        st.turn *= -1
    side = side_action_kwargs(nav_side(step, st, p))
    return action(speed=1, fw=1, **side)


BASE_P = {
    "fs": 2,
    "max_steps": 1050,
    "period": 99999,
    "forward": 99999,
    "turn": 0,
    "arc_mod": 1,
    "mode": "periodic",
    "init_turn": 1,
    "aim_k": 0.18,
    "item_k": 0.18,
    "max_turn": 8.0,
    "far_max_turn": 10.0,
    "far_turn_dx": 1e9,
    "scan_turn": 6.0,
    "arc_turn": 1.8,
    "shoot_tol": 4.0,
    "close_area": 380,
    "close_h": 42,
    "center_move": "strafe",
    "center_fw_area": 120,
    "center_fw_h": 24,
    "track_mode": "forward",
    "track_close_area": 220,
    "track_close_h": 29,
    "track_hp": 0,
    "track_hp_min_ammo": 0,
    "track_panic": False,
    "track_attack": False,
    "track_attack_ammo": 30,
    "track_attack_dx": 12.0,
    "track_attack_area": 40,
    "strafe_mode": "alternate",
    "strafe_period": 8,
    "hit_lock": 10,
    "seen_lock": 4,
    "bad_limit": 4,
    "panic": 12,
    "ammo_seek": 10,
    "hp_seek": -1,
    "urgent_item_before_enemy": False,
    "urgent_ammo": 5,
    "urgent_hp": -1,
    "urgent_item_ignore_enemy_ammo": 0,
    "urgent_enemy_dx": 18.0,
    "urgent_enemy_area": 120,
    "urgent_enemy_h": 28,
    "stuck_diff": 0.9,
    "stuck_limit": 10,
    "stuck_before_item": False,
    "stuck_random_bounce": False,
    "turn180_stuck": False,
    "turn180_bored": True,
    "bored_after": 190,
    "bored_turn": 20,
    "bored_forward": 90,
    "bored_scan_turn": 6.0,
    "bored_arc_turn": 1.8,
    "bored_context_arc": True,
    "bored_context_damage": 80,
    "bored_context_ammo": 20,
    "bored_context_hp": 85,
    "bored_context_arc_turn": 0.8,
    "adapt_arc": True,
    "adapt_once": True,
    "adapt_flip_turn": False,
    "adapt_step": 250,
    "adapt_low_damage": 15,
    "adapt_high_hp": 90,
    "adapt_low_hp": 75,
    "adapt_min_damage": 60,
    "adapt_arc_turn": 1.4,
    "adapt_bored_after": 180,
    "adapt_bored_turn": 20,
    "adapt_fast": False,
    "adapt_fast_hp": 85,
    "adapt_fast_damage": 60,
    "adapt_fast_arc_turn": 2.4,
    "adapt_fast_bored_after": 220,
    "adapt_fast_bored_turn": 20,
    "adapt_close": False,
    "adapt_close_once": True,
    "adapt_close_step": 300,
    "adapt_close_hp": 90,
    "adapt_close_ammo": 45,
    "adapt_close_damage_min": 50,
    "adapt_close_damage_max": 80,
    "adapt_close_area": 260,
    "adapt_close_h": 34,
    "nav_mode": "arc",
    "nav_y1": 52,
    "nav_y2": 155,
    "nav_dark": 32,
    "nav_min_area": 120,
    "nav_min_h": 18,
    "nav_min_w": 10,
    "nav_center_bias": 85,
    "nav_k": 0.12,
    "nav_max_turn": 5.0,
    "nav_center_tol": 10.0,
    "open_when_bored": False,
    "open_after": 140,
    "open_k": 0.12,
    "open_max_turn": 5.0,
    "open_center_tol": 12.0,
    "open_commit_steps": 0,
    "nav_strafe": "none",
    "nav_strafe_period": 24,
    "bounce_diff": 0.9,
    "bounce_bump_limit": 5,
    "bounce_steps": 8,
    "bounce_back_steps": 3,
    "bounce_turn_min": 3.0,
    "bounce_turn_max": 10.0,
    "bounce_exit_turn_scale": 0.35,
    "bounce_arc_turn": 0.25,
    "bounce_micro_jitter": 0.0,
    "bounce_turn180": False,
    "telegraph_turn_init": 1.8,
    "telegraph_min_steps": 35,
    "telegraph_max_steps": 140,
    "telegraph_turn_min": 0.2,
    "telegraph_turn_max": 2.4,
    "telegraph_bump_diff": 0.9,
    "telegraph_bump_limit": 6,
    "novelty_bounce": True,
    "novelty_w": 8,
    "novelty_h": 6,
    "novelty_memory": 80,
    "novelty_after": 160,
    "novelty_hamming": 0,
    "wall_avoid": False,
    "wall_bored": False,
    "wall_back": False,
    "wall_y1": 95,
    "wall_y2": 185,
    "wall_center_w": 35,
    "wall_side_w": 85,
    "wall_dark": 28,
    "wall_blocked_dark_max": 0.08,
    "wall_side_delta": 0.02,
    "wall_turn": 6.0,
    "close_wall_escape": False,
    "escape_y1": 72,
    "escape_y2": 178,
    "escape_center_w": 44,
    "escape_side_w": 96,
    "escape_dark": 40,
    "escape_side_dark": 38,
    "escape_mean_min": 42,
    "escape_center_dark_max": 0.24,
    "escape_edge_delta": 8,
    "escape_edge_max": 0.045,
    "escape_side_open_min": 0.18,
    "escape_side_delta": 0.08,
    "escape_turn": 7.0,
    "escape_after_progress": 120,
    "escape_steps": 8,
    "escape_forward_steps": 3,
    "escape_exit_turn_scale": 0.35,
    "escape_turn180": False,
    "c0": 50,
    "margin": 25,
    "enemy_channel": 0,
    "enemy_mode": "crcgcb",
    "rgb_r": 55,
    "rgb_rg_margin": 12,
    "rgb_rb_margin": 18,
    "rgb_g_max": 120,
    "rgb_b_max": 90,
    "min_area": 12,
    "reject_pickup_enemy": False,
    "pickup_enemy_y": 155,
    "pickup_enemy_area": 260,
    "pickup_enemy_h": 18,
    "enemy_y1": 40,
    "enemy_y2": 205,
    "enemy_min_aspect": 0,
    "enemy_center_bias": 80,
    "enemy_h_exp": 1.25,
    "enemy_y_weight": 1.0,
    "enemy_y_ref": 90,
    "enemy_body_mode": False,
    "body_c0": 42,
    "body_c2_margin": 14,
    "body_c1_max": 78,
    "body_c2_max": 60,
    "item_diff": 24,
    "item_mode": "bright",
    "item_b0": 115,
    "item_b1": 75,
    "item_b2": 70,
    "item_loose_ammo": 5,
    "item_loose_hp": -1,
    "item_loose_b0": 95,
    "clip_b0": 90,
    "clip_b1": 60,
    "clip_delta": 25,
    "item_min_area": 10,
    "item_max_area": 3000,
    "item_max_w": 60,
    "item_max_h": 52,
    "item_bright_fallback": True,
    "avoid_threat": False,
    "avoid_hp": 70,
    "threat_b0": 100,
    "threat_b1": 95,
    "threat_b2": 75,
    "threat_area": 650,
    "threat_w": 22,
    "threat_h": 22,
    "threat_max_w": 180,
    "threat_max_h": 140,
    "threat_turn_k": 0.08,
    "projectile_target": False,
    "projectile_scan": False,
    "projectile_scan_dodge": False,
    "projectile_scan_back_hp": 60,
    "proj_b0": 100,
    "proj_b1": 95,
    "proj_b2": 75,
    "proj_min_area": 10,
    "proj_max_area": 900,
    "proj_max_w": 80,
    "proj_max_h": 80,
    "proj_turn_k": 0.16,
    "proj_shoot_tol": 14.0,
    "kite": False,
    "noise_period": 0,
    "noise_ammo": 30,
}


def run(p):
    cfg = cfgfile(p.get("width", 640), p.get("height", 480), p.get("fmt", "CRCGCB"))
    env = make_gym(
        "D3Battle-v1",
        num_envs=10,
        seed=0,
        cfg_path=cfg,
        wad_path=os.path.join(BASE, "D3_battle.wad"),
        use_combined_action=False,
        stack_num=1,
        frame_skip=p["fs"],
        max_episode_steps=p["max_steps"],
        img_width=p.get("width", 640),
        img_height=p.get("height", 480),
        reward_config={"DAMAGECOUNT": [1, 0], "KILLCOUNT": [10, 0]},
        selected_weapon_reward_config={},
        game_args=p.get("game_args", ""),
        force_speed=p.get("force_speed", False),
    )
    try:
        obs, info = env.reset()
        ids = np.arange(10)
        reward = np.zeros(10)
        length = np.zeros(10)
        states = [State(p) for _ in range(10)]
        active_info = info
        final_info = [{} for _ in range(10)]
        for step in range(p["max_steps"]):
            acts = []
            for row, env_id in enumerate(ids):
                row_info = {
                    key: np.asarray(value)[row]
                    for key, value in active_info.items()
                    if np.asarray(value).ndim > 0 and len(np.asarray(value)) == len(ids)
                }
                final_info[int(env_id)] = row_info
                acts.append(choose(obs[row], row_info, states[int(env_id)], step, p))
            obs2, rew, term, trunc, info = env.step(np.asarray(acts), ids)
            done = np.logical_or(term, trunc)
            cur_ids = np.asarray(info["env_id"])
            reward[cur_ids] += rew
            length[cur_ids] += 1
            keep = ~done
            ids = cur_ids[keep]
            obs = obs2[keep]
            active_info = {
                key: np.asarray(value)[keep]
                for key, value in info.items()
                if np.asarray(value).ndim > 0 and len(np.asarray(value)) == len(done)
            }
            if len(ids) == 0:
                break
        return reward, length, final_info
    finally:
        env.close()
        os.remove(cfg)


def main():
    reward, length, finfo = run(BASE_P.copy())
    info = [
        {k: float(v) for k, v in row.items() if k in ("AMMO2", "HEALTH", "DAMAGECOUNT", "HITCOUNT", "KILLCOUNT", "DEATHCOUNT")}
        for row in finfo
    ]
    print(
        "mean",
        float(reward.mean()),
        "min",
        float(reward.min()),
        "reward",
        reward.tolist(),
        "length",
        length.astype(int).tolist(),
        "info",
        info,
        flush=True,
    )


if __name__ == "__main__":
    main()
