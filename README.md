# Learning Beyond Gradients Blog Artifacts

This repository contains the public artifacts for:

**Learning Beyond Gradients**

Published article:

- https://trinkle23897.github.io/learning-beyond-gradients/

Artifact repository:

- https://github.com/Trinkle23897/learning-beyond-gradients

The article is bilingual. The rendered HTML defaults to English and includes a Chinese switcher.

## Source Files

- `learning-beyond-gradient.en.md`: English article source.
- `learning-beyond-gradient.md`: Chinese article source.
- `learning-beyond-gradient.html`: rendered bilingual HTML.
- `render_learning_beyond_gradient.py`: local renderer.

The deployed article is `learning-beyond-gradient.html`.

## Local Preview

From the repository root:

```bash
python3 -m http.server 8000
```

Then open:

```text
http://127.0.0.1:8000/learning-beyond-gradient.html
```

Opening the HTML file directly also works in most browsers, but using `http.server` is closer to how the page is served.

## Re-render The HTML

Install the rendering dependency:

```bash
python3 -m pip install -r requirements.txt
```

Then run:

```bash
python3 render_learning_beyond_gradient.py
```

The renderer reads the English and Chinese Markdown files and rewrites `learning-beyond-gradient.html` in place.

## GitHub Pages

The site is deployed by `.github/workflows/deploy-pages.yml` on every push to `main`.

The workflow does not publish the whole repository as the website root. It builds a small `_site` directory containing:

- `index.html`, copied from `learning-beyond-gradient.html`.
- `.nojekyll`.
- Local files referenced by the article through `src` or `href`, such as figures, videos, scripts, CSVs, and prompt files.

## Included Artifacts

The repository includes the files needed to inspect and reproduce the article's representative results:

- Pong policy script.
- Breakout policy, trial summaries, sample-efficiency figure, and checkpoint videos.
- Ant policy, minimal extracted Ant policy, trial summaries, MuJoCo XML, sample-efficiency figure, and final-policy video.
- HalfCheetah policy script and iteration log.
- Montezuma exploratory policies, state/archive search scripts, summaries, probe images, plus the recovered Atari57 400-point native-image policy and replay video.
- Atari57 aggregate/per-game figures and CSV summaries.
- The Atari57 batch prompt template used for unattended Codex CLI runs.

The article appendix contains reproduction commands for five representative results. Those commands assume they are run from the repository root after cloning this repo.

## Runtime Notes

The experiments were written against EnvPool `1.1.1`. The article commands assume the relevant Python environment already has EnvPool and the Atari/MuJoCo runtime dependencies installed.

For Ant, `ant_envpool.xml` must stay next to `heuristic_ant.py`, because the reproduction command references it as:

```bash
--mujoco-xml-path ant_envpool.xml
```

## Montezuma 400-Point Replay

The Atari57 batch found one `MontezumaRevenge-v5` native-image run that reached `400.0` points. The route is packaged as a standalone open-loop replay:

```bash
python3 heuristic_montezuma_400_policy.py \
  --metadata-out heuristic_montezuma_400_replay_result.json
```

Expected result:

```text
score = 400.0
env_steps = 1769
seed = 10001
```

To regenerate the video:

```bash
python3 heuristic_montezuma_400_policy.py \
  --record-mp4 montezuma_400_render_seed10001.mp4 \
  --frame0-png montezuma_400_render_seed10001_frame0.png \
  --metadata-out montezuma_400_render_seed10001_meta.json
```

## Citation

```bibtex
@misc{weng2026learning_beyond_gradients,
  title = {Learning Beyond Gradients},
  author = {Weng, Jiayi},
  year = {2026},
  month = may,
  howpublished = {\url{https://trinkle23897.github.io/learning-beyond-gradients/}},
  note = {Blog post}
}
```
