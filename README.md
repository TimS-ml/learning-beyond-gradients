# EnvPool Heuristics Blog Artifacts

This repository contains the public artifacts for:

**Make Heuristics Great Again: Letting Codex Build Heuristic Systems from Scratch**

The rendered article is:

- `blog_heuristic_policy_atari_mujoco.html`

The source articles are:

- `blog_heuristic_policy_atari_mujoco.en.md`
- `blog_heuristic_policy_atari_mujoco.md`

The HTML page includes an English/Chinese language switcher and uses relative links to the figures, videos, scripts, and CSV files in this repository.

## Local Preview

From the repository root:

```bash
python3 -m http.server 8000
```

Then open:

```text
http://127.0.0.1:8000/blog_heuristic_policy_atari_mujoco.html
```

Opening the HTML file directly also works in most browsers, but using `http.server` is closer to how the page is served.

## Re-render the HTML

Install the only rendering dependency:

```bash
python3 -m pip install -r requirements.txt
```

Then run:

```bash
python3 render_blog.py
```

`render_blog.py` reads the two Markdown files, preserves the existing CSS and language-switching JavaScript from `blog_heuristic_policy_atari_mujoco.html`, and rewrites the HTML page in place.

## Included Artifacts

The repository includes the files needed by the article:

- Pong policy script.
- Breakout policy, trial logs, sample-efficiency figure, and checkpoint videos.
- Ant policy, minimal extracted Ant policy, trial logs, sample-efficiency figure, MuJoCo XML, and final-policy video.
- HalfCheetah policy script, iteration log, and sample-efficiency figure.
- Montezuma exploratory policies, state/archive search scripts, trial logs, summaries, and probe images.
- Atari57 aggregate and per-game figures, plus the CSV files used to summarize the aggregate/per-game comparisons.
- The Atari57 batch prompt template used for the unattended Codex CLI runs.

The reproduction commands for individual Breakout and Ant checkpoints are embedded in collapsible sections inside the article. Those commands assume they are run from the repository root.

HalfCheetah's longer iteration history and checked policy commands are in `heuristic_halfcheetah_v5_log.md`; the corresponding script is `heuristic_halfcheetah_v5.py`.

## Runtime Notes

The experiments were written against EnvPool `1.1.1`. The article commands assume the relevant Python environment already has EnvPool and the Atari/MuJoCo runtime dependencies installed.

For Ant, `ant_envpool.xml` must stay next to `heuristic_ant.py`, because the reproduction command references it as:

```bash
--mujoco-xml-path ant_envpool.xml
```

## Citation

```bibtex
@misc{weng2026codex_heuristic_policy,
  title = {Make Heuristics Great Again: Letting Codex Build Heuristic Systems from Scratch},
  author = {Weng, Jiayi},
  year = {2026},
  month = apr,
  howpublished = {\url{https://example.com/codex-heuristic-policy}},
  note = {Blog post}
}
```
