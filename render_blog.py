from pathlib import Path
import re

import markdown


ROOT = Path(__file__).resolve().parent
HTML_PATH = ROOT / "blog_heuristic_policy_atari_mujoco.html"
EN_MD = ROOT / "blog_heuristic_policy_atari_mujoco.en.md"
ZH_MD = ROOT / "blog_heuristic_policy_atari_mujoco.md"


def render_markdown(path: Path) -> str:
    return markdown.markdown(
        path.read_text(),
        extensions=["extra", "fenced_code", "tables", "sane_lists"],
        output_format="html5",
    )


def extract_block(pattern: str, html: str, name: str) -> str:
    match = re.search(pattern, html, re.S)
    if match is None:
        raise RuntimeError(f"Could not find {name} block in {HTML_PATH}")
    return match.group(1)


def main() -> None:
    current = HTML_PATH.read_text()
    style = extract_block(r"<style>\n?(.*?)\n?  </style>", current, "style")
    script = extract_block(r"<script>\n?(.*?)\n?  </script>", current, "script")

    en_html = render_markdown(EN_MD)
    zh_html = render_markdown(ZH_MD)

    HTML_PATH.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Make Heuristics Great Again: Letting Codex Build Heuristic Systems from Scratch</title>
  <style>
{style}
  </style>
</head>
<body>
  <main class="page">
    <div class="lang-switch-wrap" aria-label="Language switcher">
      <div class="lang-switch">
        <button type="button" id="lang-en" class="active" aria-pressed="true">English</button>
        <button type="button" id="lang-zh" aria-pressed="false">中文</button>
      </div>
    </div>
    <article id="article-en" class="lang-pane" lang="en">
{en_html}
    </article>
    <article id="article-zh" class="lang-pane" lang="zh-CN" hidden>
{zh_html}
    </article>
  </main>
  <script>
{script}
  </script>
</body>
</html>
"""
    )


if __name__ == "__main__":
    main()
