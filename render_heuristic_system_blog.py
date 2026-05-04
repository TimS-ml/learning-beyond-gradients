from pathlib import Path
import re

import markdown


ROOT = Path(__file__).resolve().parent
SOURCE_MD = ROOT / "blog_heuristic_system.md"
HTML_PATH = ROOT / "blog_heuristic_system.html"
STYLE_SOURCE = ROOT / "blog_heuristic_policy_atari_mujoco.html"


def extract_style(html: str) -> str:
    match = re.search(r"<style>\n?(.*?)\n?  </style>", html, re.S)
    if match is None:
        raise RuntimeError(f"Could not find style block in {STYLE_SOURCE}")
    return match.group(1)


def render_markdown(path: Path) -> str:
    return markdown.markdown(
        path.read_text(),
        extensions=["extra", "fenced_code", "tables", "sane_lists"],
        output_format="html5",
    )


def main() -> None:
    style = extract_style(STYLE_SOURCE.read_text())
    article_html = render_markdown(SOURCE_MD)
    HTML_PATH.write_text(
        f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>heuristic system</title>
  <style>
{style}
  </style>
</head>
<body>
  <main class="page">
    <article class="lang-pane" lang="zh-CN">
{article_html}
    </article>
  </main>
</body>
</html>
"""
    )


if __name__ == "__main__":
    main()
