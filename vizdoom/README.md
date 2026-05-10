# VizDoom Heuristic Policies

This folder contains the VizDoom smoke-test policies used in the article
appendix.  They are meant as reproducible EnvPool heuristics, not as learned
agents.

## D3 Battle: Screen-CV Policy

Entrypoint:

```bash
python vizdoom/heuristic_vizdoom_d3_cv.py
```

Local 10-seed result with `seed=0`, `frame_skip=2`, and reward
`DAMAGECOUNT + 10 * KILLCOUNT`:

```text
mean = 557.0
min = 440.0
rewards = [545, 475, 480, 440, 690, 500, 600, 595, 530, 715]
```

The D3 policy uses the rendered observation plus public EnvPool game variables
such as health, ammo, hit count, and damage count.  It does not read the WAD
map, object coordinates, labels, or seed-specific routes.

To regenerate the 35fps 10-seed render grid:

```bash
python vizdoom/record_vizdoom_d3_cv.py
```

Video artifact: `d3_cv_best_10seed_render_35fps.mp4`.

## D1 Basic: Pure EnvPool CV Medikit Policy

Entrypoint:

```bash
python vizdoom/heuristic_vizdoom_d1_cv.py --episodes 10 --seed 0
```

Local 10-seed result:

```text
mean = 0.9440999741666019
min = 0.28999998047947884
rewards = [
  1.0799999684095383,
  1.0399999842047691,
  0.28999998047947884,
  0.9279999658465385,
  1.0799999684095383,
  1.0099999718368053,
  1.0349999852478504,
  1.004999976605177,
  0.9439999675378203,
  1.0289999730885029,
]
```

The D1 policy uses only pip-installable EnvPool, rendered screen pixels, and
the public `HEALTH` game variable.  It does not read the map, object
coordinates, labels, a locally compiled EnvPool extension, or seed-specific
routes.  It stages near the medikit while health is high, then walks into the
medikit once the pickup has reward value.

To regenerate the 35fps 10-seed render grid:

```bash
python vizdoom/heuristic_vizdoom_d1_cv.py \
  --episodes 10 \
  --seed 0 \
  --record-mp4 vizdoom/d1_cv_10seed_render_35fps.mp4
```

Video artifact: `d1_cv_10seed_render_35fps.mp4`.
