from pathlib import Path
import argparse
import re
from html import escape, unescape

import markdown


ROOT = Path(__file__).resolve().parent
EN_MD = ROOT / "learning-beyond-gradient.en.md"
ZH_MD = ROOT / "learning-beyond-gradient.md"
HTML_PATH = ROOT / "learning-beyond-gradient.html"
PAGE_TITLE_EN = "Learning Beyond Gradients"
PAGE_URL = "https://trinkle23897.github.io/learning-beyond-gradients/"
SOCIAL_IMAGE_URL = f"{PAGE_URL}ig_0c2dd0d2f07176560169fbc256930481969d3c6ba3316d5486.png"
SOCIAL_IMAGE_WIDTH = 1672
SOCIAL_IMAGE_HEIGHT = 941


STYLE = """:root {
  color-scheme: light;
  --bg: #f8fafc;
  --paper: #ffffff;
  --ink: #111827;
  --muted: #4b5563;
  --line: #d9e2ec;
  --accent: #0f766e;
  --accent-2: #b45309;
  --code-bg: #0f172a;
  --code-ink: #e5e7eb;
  --inline-code-bg: #edf2f7;
  --toc-bg: #f8fafc;
  --switch-bg: rgba(255, 255, 255, 0.94);
  --active-bg: #111827;
  --active-ink: #ffffff;
  --details-bg: #fbfdff;
  --summary-bg: #fff7ed;
  --quote-bg: #f0fdfa;
  --table-head-bg: #f1f5f9;
  --media-bg: #ffffff;
  --page-shadow: 0 24px 80px rgba(15, 23, 42, 0.08);
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --bg: #0b1120;
  --paper: #111827;
  --ink: #e5e7eb;
  --muted: #a7b0c0;
  --line: #2d3748;
  --accent: #5eead4;
  --accent-2: #fbbf24;
  --code-bg: #020617;
  --code-ink: #e5e7eb;
  --inline-code-bg: #1f2937;
  --toc-bg: #0f172a;
  --switch-bg: rgba(17, 24, 39, 0.94);
  --active-bg: #e5e7eb;
  --active-ink: #111827;
  --details-bg: #0f172a;
  --summary-bg: #1f2937;
  --quote-bg: #092f2d;
  --table-head-bg: #172033;
  --media-bg: #ffffff;
  --page-shadow: 0 24px 80px rgba(0, 0, 0, 0.32);
}
* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--ink);
  font-family: ui-serif, Georgia, Cambria, "Times New Roman", serif;
  font-size: 18px;
  line-height: 1.72;
}
.page {
  max-width: 980px;
  margin: 0 auto;
  padding: 56px 28px 96px;
  background: var(--paper);
  min-height: 100vh;
  box-shadow: var(--page-shadow);
}
.lang-switch-wrap {
  position: sticky;
  top: 12px;
  z-index: 20;
  display: flex;
  justify-content: flex-end;
  pointer-events: none;
  margin: -30px 0 22px;
}
.top-controls {
  pointer-events: auto;
  display: inline-flex;
  gap: 8px;
  align-items: center;
}
.lang-switch {
  pointer-events: auto;
  display: inline-flex;
  gap: 4px;
  padding: 4px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: var(--switch-bg);
  box-shadow: 0 10px 30px rgba(15, 23, 42, 0.12);
  backdrop-filter: blur(8px);
}
.lang-switch button,
.theme-toggle {
  min-width: 72px;
  border: 0;
  border-radius: 6px;
  padding: 7px 12px;
  background: transparent;
  color: var(--muted);
  font: 700 14px/1 ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  cursor: pointer;
}
.theme-toggle {
  border: 1px solid var(--line);
  background: var(--switch-bg);
  box-shadow: 0 10px 30px rgba(15, 23, 42, 0.12);
  backdrop-filter: blur(8px);
}
.lang-switch button.active {
  background: var(--active-bg);
  color: var(--active-ink);
}
.lang-pane[hidden] { display: none !important; }
h1, h2, h3, h4 {
  font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  line-height: 1.18;
  letter-spacing: 0;
}
h1 { margin: 0 0 28px; font-size: 46px; max-width: 860px; }
h2 { margin: 56px 0 18px; padding-top: 10px; border-top: 1px solid var(--line); font-size: 30px; }
h3 { margin: 34px 0 12px; font-size: 22px; }
h4 { margin: 28px 0 10px; font-size: 19px; }
h2, h3, h4 { scroll-margin-top: 78px; }
.section-link {
  margin-right: 0.35em;
  color: var(--muted);
  font-size: 0.82em;
  font-weight: 700;
  text-decoration: none;
  opacity: 0.72;
}
.section-link:hover {
  color: var(--accent);
  opacity: 1;
  text-decoration: underline;
}
p { margin: 18px 0; }
a { color: var(--accent); text-decoration-thickness: 1px; text-underline-offset: 3px; }
ul, ol { padding-left: 1.45em; }
li { margin: 8px 0; }
.toc {
  margin: 24px 0 34px;
  padding: 16px 18px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: var(--toc-bg);
  font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
.toc-title {
  margin: 0 0 10px;
  font-weight: 800;
  color: var(--muted);
}
.toc ol { margin: 0; padding: 0; list-style: none; }
.toc li { margin: 5px 0; line-height: 1.35; }
.toc a { color: var(--ink); text-decoration: none; }
.toc a:hover { color: var(--accent); text-decoration: underline; text-underline-offset: 3px; }
.toc-level-3 { padding-left: 1.25rem; font-size: 0.93em; }
code {
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
  font-size: 0.88em;
  background: var(--inline-code-bg);
  padding: 0.1em 0.28em;
  border-radius: 4px;
}
pre { overflow-x: auto; margin: 22px 0; padding: 18px 20px; border-radius: 8px; background: var(--code-bg); color: var(--code-ink); line-height: 1.48; }
pre code { background: transparent; color: inherit; padding: 0; border-radius: 0; font-size: 0.88em; }
img, video { display: block; max-width: 100%; height: auto; margin: 26px auto; border: 1px solid var(--line); border-radius: 8px; background: var(--media-bg); }
img[src="ig_0c2dd0d2f07176560169fbc256930481969d3c6ba3316d5486.png"] { transition: filter 160ms ease, box-shadow 160ms ease; }
:root[data-theme="dark"] img[src="ig_0c2dd0d2f07176560169fbc256930481969d3c6ba3316d5486.png"] {
  filter: invert(1) hue-rotate(180deg) saturate(0.9) brightness(1.04);
}
video { background: #0b1020; }
details { margin: 24px 0 30px; border: 1px solid var(--line); border-radius: 8px; background: var(--details-bg); overflow: hidden; }
summary { cursor: pointer; padding: 14px 18px; font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; font-weight: 700; color: var(--accent-2); background: var(--summary-bg); border-bottom: 1px solid var(--line); }
details > *:not(summary) { margin-left: 18px; margin-right: 18px; }
details pre { margin: 18px; max-height: 70vh; }
blockquote { margin: 24px 0; padding: 2px 20px; border-left: 4px solid var(--accent); color: var(--muted); background: var(--quote-bg); }
table { border-collapse: collapse; width: 100%; margin: 24px 0; font-size: 0.94em; }
th, td { border: 1px solid var(--line); padding: 8px 10px; text-align: left; }
th { background: var(--table-head-bg); }
.github-issue-comments {
  margin: 28px 0 12px;
  padding: 0;
  font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
.github-comments-head {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  align-items: start;
  gap: 16px;
  padding: 0 0 13px;
  border-bottom: 2px solid var(--line);
  margin-bottom: 16px;
}
.github-comments-head > div:first-child {
  min-width: 0;
}
.github-comments-count {
  margin: 0;
  color: var(--ink);
  font-size: 1.05em;
  font-weight: 800;
  line-height: 1.25;
}
.github-comments-subtitle,
.github-comments-powered {
  margin: 5px 0 0;
  color: var(--muted);
  font-size: 0.82em;
  line-height: 1.45;
}
.github-comments-controls {
  display: flex;
  flex-wrap: nowrap;
  justify-content: flex-end;
  gap: 8px;
  align-items: center;
  min-width: max-content;
}
.github-comments-new {
  flex: 0 0 auto;
  display: inline-flex;
  align-items: center;
  border: 1px solid var(--line);
  border-radius: 6px;
  padding: 8px 12px;
  color: var(--active-ink);
  background: var(--active-bg);
  font-size: 0.8em;
  font-weight: 700;
  line-height: 1.1;
  text-decoration: none;
}
.github-comments-new:hover {
  color: var(--active-ink);
  opacity: 0.9;
  text-decoration: none;
}
.github-comments-status,
.github-comments-empty {
  margin: 14px 0 0;
  padding: 18px 0;
  border-bottom: 1px solid var(--line);
  color: var(--muted);
  font-size: 0.9em;
}
.github-comments-list {
  display: grid;
  gap: 0;
  margin-top: 4px;
}
.github-comment {
  display: grid;
  grid-template-columns: 42px 1fr;
  gap: 13px;
  padding: 18px 0;
  border-bottom: 1px solid var(--line);
}
.github-comment-avatar {
  width: 42px;
  height: 42px;
  border-radius: 50%;
  border: 1px solid var(--line);
  background: var(--toc-bg);
  object-fit: cover;
  margin: 1px 0 0;
}
.github-comment-main {
  min-width: 0;
}
.github-comment-byline {
  display: flex;
  flex-wrap: wrap;
  gap: 7px;
  align-items: baseline;
  color: var(--muted);
  font-size: 0.82em;
  line-height: 1.35;
}
.github-comment-author {
  color: var(--ink);
  font-weight: 800;
  text-decoration: none;
}
.github-comment-author:hover {
  color: var(--accent);
  text-decoration: underline;
}
.github-comment-title {
  margin: 5px 0 0;
  font-size: 1.02em;
  line-height: 1.35;
}
.github-comment-title a {
  color: var(--ink);
  text-decoration: none;
}
.github-comment-title a:hover {
  color: var(--accent);
  text-decoration: underline;
}
.github-comment-body {
  margin: 9px 0 0;
  color: var(--ink);
  font-size: 0.92em;
  line-height: 1.55;
  white-space: pre-wrap;
  overflow-wrap: anywhere;
}
.github-comment-actions {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
  align-items: center;
  margin-top: 10px;
  font-size: 0.8em;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.02em;
}
.github-comment-actions a,
.github-comment-actions span {
  color: var(--muted);
  text-decoration: none;
}
.github-comment-actions a:hover {
  color: var(--accent);
  text-decoration: underline;
}
.github-replies {
  display: grid;
  gap: 0;
  margin-top: 14px;
  padding-left: 14px;
  border-left: 2px solid var(--line);
}
.github-reply {
  display: grid;
  grid-template-columns: 30px 1fr;
  gap: 10px;
  padding: 12px 0 0;
}
.github-reply-avatar {
  width: 30px;
  height: 30px;
  border-radius: 50%;
  border: 1px solid var(--line);
  background: var(--toc-bg);
  object-fit: cover;
  margin: 2px 0 0;
}
.github-reply-main {
  min-width: 0;
}
.github-reply-byline {
  display: flex;
  flex-wrap: wrap;
  gap: 7px;
  align-items: baseline;
  color: var(--muted);
  font-size: 0.8em;
  line-height: 1.35;
}
.github-reply-author {
  color: var(--ink);
  font-weight: 800;
  text-decoration: none;
}
.github-reply-author:hover {
  color: var(--accent);
  text-decoration: underline;
}
.github-reply-body {
  margin: 6px 0 0;
  color: var(--ink);
  font-size: 0.88em;
  line-height: 1.52;
  white-space: pre-wrap;
  overflow-wrap: anywhere;
}
.github-replies-error {
  margin: 10px 0 0;
  color: var(--muted);
  font-size: 0.82em;
}
@media (max-width: 720px) {
  body { font-size: 16px; }
  .page { padding: 32px 18px 64px; box-shadow: none; }
  .lang-switch-wrap { margin: 0 0 18px; top: 0; }
  .lang-switch button { min-width: 66px; padding: 7px 10px; }
  .theme-toggle { min-width: 62px; padding: 7px 10px; }
  .github-comments-head { display: block; }
  .github-comments-controls { justify-content: flex-start; margin-top: 10px; min-width: 0; }
  .github-comments-new { margin-top: 0; }
  .github-comment { grid-template-columns: 34px 1fr; gap: 10px; }
  .github-comment-avatar { width: 34px; height: 34px; }
  .github-replies { padding-left: 10px; }
  .github-reply { grid-template-columns: 26px 1fr; gap: 8px; }
  .github-reply-avatar { width: 26px; height: 26px; }
  h1 { font-size: 34px; }
  h2 { font-size: 24px; }
}
"""


THEME_BOOTSTRAP = """(function () {
  const storageKey = 'learning_beyond_gradient_theme_v1';
  let theme = null;
  try {
    const saved = window.localStorage.getItem(storageKey);
    if (saved === 'light' || saved === 'dark') theme = saved;
  } catch (_) {}
  if (!theme && window.matchMedia) {
    theme = window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  }
  document.documentElement.dataset.theme = theme || 'light';
})();
"""


LANG_SCRIPT = """(function () {
  const panes = {
    en: document.getElementById('article-en'),
    zh: document.getElementById('article-zh'),
  };
  const buttons = {
    en: document.getElementById('lang-en'),
    zh: document.getElementById('lang-zh'),
  };
  const titles = {
    en: 'Learning Beyond Gradients',
    zh: 'Learning Beyond Gradients',
  };
  const storageKey = 'learning_beyond_gradient_lang_v1';
  const themeStorageKey = 'learning_beyond_gradient_theme_v1';
  const themeButton = document.getElementById('theme-toggle');

  function preferredTheme() {
    return window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  }

  function savedTheme() {
    try {
      const saved = window.localStorage.getItem(themeStorageKey);
      if (saved === 'light' || saved === 'dark') return saved;
    } catch (_) {}
    return null;
  }

  function currentTheme() {
    return savedTheme() || preferredTheme();
  }

  function setTheme(theme, persist) {
    document.documentElement.dataset.theme = theme;
    if (themeButton) {
      themeButton.textContent = theme === 'dark' ? 'Light' : 'Dark';
      themeButton.setAttribute('aria-pressed', theme === 'dark' ? 'true' : 'false');
      themeButton.setAttribute('aria-label', theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode');
    }
    if (persist) {
      try { window.localStorage.setItem(themeStorageKey, theme); } catch (_) {}
    }
  }

  function setLanguage(lang, updateUrl) {
    if (!panes[lang]) lang = 'en';
    for (const key of Object.keys(panes)) {
      const selected = key === lang;
      panes[key].hidden = !selected;
      buttons[key].classList.toggle('active', selected);
      buttons[key].setAttribute('aria-pressed', selected ? 'true' : 'false');
    }
    document.documentElement.lang = lang === 'zh' ? 'zh-CN' : 'en';
    document.title = titles[lang];
    try { window.localStorage.setItem(storageKey, lang); } catch (_) {}
    if (updateUrl) {
      const url = new URL(window.location.href);
      url.hash = lang === 'zh' ? 'zh' : '';
      window.history.replaceState(null, '', url);
    }
  }

  buttons.en.addEventListener('click', () => setLanguage('en', true));
  buttons.zh.addEventListener('click', () => setLanguage('zh', true));
  if (themeButton) {
    themeButton.addEventListener('click', () => {
      setTheme(currentTheme() === 'dark' ? 'light' : 'dark', true);
    });
  }

  let initial = 'en';
  if (window.location.hash === '#zh' || window.location.hash.startsWith('#zh-')) {
    initial = 'zh';
  } else if (window.location.hash === '#en' || window.location.hash.startsWith('#en-')) {
    initial = 'en';
  } else {
    try {
      const saved = window.localStorage.getItem(storageKey);
      if (saved === 'en' || saved === 'zh') initial = saved;
    } catch (_) {}
  }
  setLanguage(initial, false);
  setTheme(currentTheme(), false);
  if (window.matchMedia) {
    window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => {
      if (!savedTheme()) setTheme(preferredTheme(), false);
    });
  }
})();
"""


COMMENTS_SCRIPT = """(function () {
  const containers = Array.from(document.querySelectorAll('.github-issue-comments'));
  if (!containers.length || !window.fetch) return;

  const copy = {
    en: {
      title: 'Comments',
      subtitle: 'Each GitHub issue is one comment, loaded live in your browser.',
      newComment: 'Join discussion',
      loading: 'Loading comments from GitHub...',
      empty: 'No issues yet. Open one to start the thread.',
      error: 'Could not load GitHub issues right now.',
      reply: 'Reply',
      viewThread: 'View thread',
      replyLink: 'permalink',
      powered: 'Powered by GitHub Issues',
      replies: 'replies',
      repliesError: 'Could not load replies.',
    },
    zh: {
      title: '评论',
      subtitle: '每个 GitHub issue 是一条评论，页面打开时动态加载。',
      newComment: '参与讨论',
      loading: '正在从 GitHub 加载评论...',
      empty: '还没有 issue。开一个就会出现在这里。',
      error: '现在没能加载 GitHub Issues。',
      reply: '回复',
      viewThread: '查看讨论',
      replyLink: '链接',
      powered: '由 GitHub Issues 驱动',
      replies: '条回复',
      repliesError: '没能加载这条下面的回复。',
    },
  };
  const issueCommentRequests = new Map();

  function t(lang) {
    return copy[lang === 'zh' ? 'zh' : 'en'];
  }

  function escapeHtml(value) {
    return String(value || '')
      .replaceAll('&', '&amp;')
      .replaceAll('<', '&lt;')
      .replaceAll('>', '&gt;')
      .replaceAll('"', '&quot;')
      .replaceAll("'", '&#39;');
  }

  function linkify(text) {
    return escapeHtml(text).replace(
      /(https?:\\/\\/[^\\s<]+)/g,
      '<a href="$1">$1</a>'
    );
  }

  function formatDate(value, lang) {
    try {
      return new Intl.DateTimeFormat(lang === 'zh' ? 'zh-CN' : 'en', {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
      }).format(new Date(value));
    } catch (_) {
      return value ? value.slice(0, 10) : '';
    }
  }

  function issueBody(issue) {
    const body = (issue.body || '').trim();
    if (!body) return '';
    return body.length > 900 ? `${body.slice(0, 900).trim()}...` : body;
  }

  function threadBody(comment) {
    const body = (comment.body || '').trim();
    if (!body) return '';
    return body.length > 700 ? `${body.slice(0, 700).trim()}...` : body;
  }

  function countLabel(count, lang) {
    if (count === null) return t(lang).title;
    if (lang === 'zh') return `${count} 条评论`;
    return count === 1 ? '1 Comment' : `${count} Comments`;
  }

  function headerHtml(container, lang, count) {
    const repo = container.dataset.repo || 'Trinkle23897/learning-beyond-gradients';
    const issueUrl = container.dataset.issueUrl || `https://github.com/${repo}/issues/new?title=Comment%3A%20Learning%20Beyond%20Gradients`;
    const text = t(lang);
    return `<div class="github-comments-head">
      <div>
        <p class="github-comments-count">${escapeHtml(countLabel(count, lang))}</p>
        <p class="github-comments-subtitle">${escapeHtml(text.subtitle)}</p>
        <p class="github-comments-powered">${escapeHtml(text.powered)}</p>
      </div>
      <div class="github-comments-controls">
        <a class="github-comments-new" href="${escapeHtml(issueUrl)}">${escapeHtml(text.newComment)}</a>
      </div>
    </div>`;
  }

  function sortIssues(issues) {
    return [...issues].sort((a, b) => {
      return new Date(b.created_at).getTime() - new Date(a.created_at).getTime();
    });
  }

  function totalCommentCount(issues) {
    return issues.reduce((count, issue) => {
      const replies = Array.isArray(issue.threadComments) ? issue.threadComments.length : 0;
      return count + 1 + replies;
    }, 0);
  }

  function replyCountLabel(count, lang) {
    const text = t(lang);
    if (lang === 'zh') return `${count}${text.replies}`;
    return count === 1 ? '1 reply' : `${count} ${text.replies}`;
  }

  function renderReply(comment, lang) {
    const text = t(lang);
    const user = comment.user && comment.user.login ? comment.user.login : 'github-user';
    const avatarUrl = comment.user && comment.user.avatar_url ? comment.user.avatar_url : '';
    const userUrl = comment.user && comment.user.html_url ? comment.user.html_url : comment.html_url;
    const body = threadBody(comment);
    return `<div class="github-reply">
      ${avatarUrl ? `<img class="github-reply-avatar" src="${escapeHtml(avatarUrl)}" alt="${escapeHtml(user)} avatar" loading="lazy">` : '<div class="github-reply-avatar" aria-hidden="true"></div>'}
      <div class="github-reply-main">
        <div class="github-reply-byline">
          <a class="github-reply-author" href="${escapeHtml(userUrl)}">${escapeHtml(user)}</a>
          <span>${escapeHtml(formatDate(comment.created_at, lang))}</span>
          <a href="${escapeHtml(comment.html_url)}">${escapeHtml(text.replyLink)}</a>
        </div>
        ${body ? `<div class="github-reply-body">${linkify(body)}</div>` : ''}
      </div>
    </div>`;
  }

  function renderReplies(issue, lang) {
    const text = t(lang);
    const replies = Array.isArray(issue.threadComments)
      ? [...issue.threadComments].sort((a, b) => new Date(a.created_at).getTime() - new Date(b.created_at).getTime())
      : [];
    if (issue.threadCommentsError) {
      return `<div class="github-replies-error">${escapeHtml(text.repliesError)}</div>`;
    }
    if (!replies.length) return '';
    return `<div class="github-replies">
      ${replies.map((comment) => renderReply(comment, lang)).join('')}
    </div>`;
  }

  function renderIssue(issue, lang) {
    const text = t(lang);
    const user = issue.user && issue.user.login ? issue.user.login : 'github-user';
    const body = issueBody(issue);
    const replies = Number(issue.comments) || 0;
    const avatarUrl = issue.user && issue.user.avatar_url ? issue.user.avatar_url : '';
    const userUrl = issue.user && issue.user.html_url ? issue.user.html_url : issue.html_url;
    return `<article class="github-comment">
      ${avatarUrl ? `<img class="github-comment-avatar" src="${escapeHtml(avatarUrl)}" alt="${escapeHtml(user)} avatar" loading="lazy">` : '<div class="github-comment-avatar" aria-hidden="true"></div>'}
      <div class="github-comment-main">
        <div class="github-comment-byline">
          <a class="github-comment-author" href="${escapeHtml(userUrl)}">${escapeHtml(user)}</a>
          <span>#${issue.number}</span>
          <span>${escapeHtml(formatDate(issue.created_at, lang))}</span>
        </div>
        <h3 class="github-comment-title"><a href="${escapeHtml(issue.html_url)}">${escapeHtml(issue.title)}</a></h3>
        ${body ? `<div class="github-comment-body">${linkify(body)}</div>` : ''}
        <div class="github-comment-actions">
          <a href="${escapeHtml(issue.html_url)}#new_comment_field">${escapeHtml(text.reply)}</a>
          <a href="${escapeHtml(issue.html_url)}">${escapeHtml(text.viewThread)}</a>
          ${replies ? `<span>${escapeHtml(replyCountLabel(replies, lang))}</span>` : ''}
        </div>
        ${renderReplies(issue, lang)}
      </div>
    </article>`;
  }

  function render(container, issues) {
    const lang = container.dataset.lang || document.documentElement.lang || 'en';
    const text = t(lang);
    const comments = issues.filter((issue) => !issue.pull_request);
    if (!comments.length) {
      container.innerHTML = `${headerHtml(container, lang, 0)}<p class="github-comments-empty">${escapeHtml(text.empty)}</p>`;
      return;
    }
    const sortedComments = sortIssues(comments);
    container.innerHTML = `${headerHtml(container, lang, totalCommentCount(comments))}
      <div class="github-comments-list">
        ${sortedComments.map((issue) => renderIssue(issue, lang)).join('')}
      </div>`;
  }

  const requests = new Map();
  function fetchJson(url) {
    return fetch(url, {
      headers: {
        Accept: 'application/vnd.github+json',
        'X-GitHub-Api-Version': '2022-11-28',
      },
    }).then((response) => {
      if (!response.ok) throw new Error(`GitHub API ${response.status}`);
      return response.json();
    });
  }

  function issueCommentsUrl(issue) {
    const url = new URL(issue.comments_url);
    url.searchParams.set('per_page', '20');
    return url.toString();
  }

  function fetchIssueComments(issue) {
    if (!Number(issue.comments) || !issue.comments_url || issue.pull_request) {
      return Promise.resolve({ ...issue, threadComments: [] });
    }
    const url = issueCommentsUrl(issue);
    if (!issueCommentRequests.has(url)) {
      issueCommentRequests.set(url, fetchJson(url));
    }
    return issueCommentRequests.get(url)
      .then((comments) => ({
        ...issue,
        threadComments: Array.isArray(comments) ? comments : [],
      }))
      .catch(() => ({
        ...issue,
        threadComments: [],
        threadCommentsError: true,
      }));
  }

  function attachThreadComments(issues) {
    return Promise.all(issues.map((issue) => fetchIssueComments(issue)));
  }

  function load(container) {
    const repo = container.dataset.repo || 'Trinkle23897/learning-beyond-gradients';
    const lang = container.dataset.lang || document.documentElement.lang || 'en';
    const text = t(lang);
    container.innerHTML = `${headerHtml(container, lang, null)}<p class="github-comments-status">${escapeHtml(text.loading)}</p>`;
    if (!requests.has(repo)) {
      const url = `https://api.github.com/repos/${repo}/issues?state=all&sort=created&direction=desc&per_page=30`;
      requests.set(repo, fetchJson(url).then((issues) => {
        const parsedIssues = Array.isArray(issues) ? issues : [];
        return attachThreadComments(parsedIssues);
      }));
    }
    requests.get(repo)
      .then((issues) => {
        const parsedIssues = Array.isArray(issues) ? issues : [];
        render(container, parsedIssues);
      })
      .catch(() => {
        container.innerHTML = `${headerHtml(container, lang, null)}<p class="github-comments-status">${escapeHtml(text.error)}</p>`;
      });
  }

  containers.forEach(load);
})();
"""


HEADING_RE = re.compile(r"<h([234])>(.*?)</h\1>")
TAG_RE = re.compile(r"<[^>]+>")
ASCII_WORD_RE = re.compile(r"[A-Za-z0-9]+")
CJK_SLUG_HINTS = (
    ("中间节点", "intermediate-nodes"),
    ("默认", "default"),
    ("策略", "policy"),
    ("分回放", "point-replay"),
)


def heading_text(fragment: str) -> str:
    return unescape(TAG_RE.sub("", fragment)).strip()


def unique_id(base: str, used: set[str]) -> str:
    candidate = base
    suffix = 2
    while candidate in used:
        candidate = f"{base}-{suffix}"
        suffix += 1
    used.add(candidate)
    return candidate


def heading_id(text: str, lang: str, index: int, used: set[str]) -> str:
    match = re.match(r"^((?:\d+|[A-Z])(?:\.\d+)*)(?:\.|\s)", text)
    if match:
        base = f"{lang}-section-{match.group(1).lower().replace('.', '-')}"
    elif "Appendix" in text or text.startswith("附录"):
        base = f"{lang}-appendix"
    elif text in {"Disclaimer", "免责声明"}:
        base = f"{lang}-disclaimer"
    elif text in {"Acknowledgements", "致谢"}:
        base = f"{lang}-acknowledgements"
    elif text in {"Citation", "引用"}:
        base = f"{lang}-citation"
    else:
        base = f"{lang}-section-{index}"

    return unique_id(base, used)


def subsection_id(text: str, lang: str, index: int, used: set[str]) -> str:
    tokens = [match.group(0).lower() for match in ASCII_WORD_RE.finditer(text)]
    for phrase, replacement in CJK_SLUG_HINTS:
        if phrase in text:
            tokens.extend(replacement.split("-"))
    base = f"{lang}-{'-'.join(tokens)}" if tokens else f"{lang}-subsection-{index}"
    return unique_id(base, used)


def heading_with_link(level: int, inner: str, text: str, anchor: str) -> str:
    label = escape(f"Link to {text}", quote=True)
    return (
        f'<h{level} id="{anchor}">'
        f'<a class="section-link" href="#{anchor}" aria-label="{label}">#</a>'
        f"{inner}"
        f"</h{level}>"
    )


def add_heading_ids(article_html: str, lang: str) -> tuple[str, list[tuple[int, str, str]]]:
    entries: list[tuple[int, str, str]] = []
    used: set[str] = set()
    subsection_count = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal subsection_count
        level = int(match.group(1))
        inner = match.group(2)
        text = heading_text(inner)
        if level == 4:
            subsection_count += 1
            anchor = subsection_id(text, lang, subsection_count, used)
            return heading_with_link(level, inner, text, anchor)

        anchor = heading_id(text, lang, len(entries) + 1, used)
        entries.append((level, text, anchor))
        return heading_with_link(level, inner, text, anchor)

    return HEADING_RE.sub(replace, article_html), entries


def build_toc(entries: list[tuple[int, str, str]], lang: str) -> str:
    title = "目录" if lang == "zh" else "Contents"
    items = "\n".join(
        f'      <li class="toc-level-{level}"><a href="#{anchor}">{escape(text)}</a></li>'
        for level, text, anchor in entries
    )
    return f"""<nav class="toc" aria-label="{escape(title)}">
    <div class="toc-title">{escape(title)}</div>
    <ol>
{items}
    </ol>
  </nav>"""


def inject_toc(article_html: str, entries: list[tuple[int, str, str]], lang: str) -> str:
    toc = build_toc(entries, lang)
    marker = "</blockquote>"
    if marker in article_html:
        return article_html.replace(marker, f"{marker}\n{toc}", 1)
    return f"{toc}\n{article_html}"


def render_markdown(path: Path, lang: str) -> str:
    article_html = markdown.markdown(
        path.read_text(),
        extensions=["extra", "fenced_code", "tables", "sane_lists"],
        output_format="html5",
    )
    article_html, entries = add_heading_ids(article_html, lang)
    return inject_toc(article_html, entries, lang)


def social_meta(title: str) -> str:
    safe_title = escape(title, quote=True)
    safe_url = escape(PAGE_URL, quote=True)
    safe_image = escape(SOCIAL_IMAGE_URL, quote=True)
    return f"""  <meta name="description" content="{safe_title}">
  <meta itemprop="name" content="{safe_title}">
  <meta itemprop="description" content="{safe_title}">
  <meta itemprop="image" content="{safe_image}">
  <link rel="image_src" href="{safe_image}">
  <meta property="og:type" content="article">
  <meta property="og:site_name" content="{safe_title}">
  <meta property="og:title" content="{safe_title}">
  <meta property="og:description" content="{safe_title}">
  <meta property="og:url" content="{safe_url}">
  <meta property="og:image" content="{safe_image}">
  <meta property="og:image:width" content="{SOCIAL_IMAGE_WIDTH}">
  <meta property="og:image:height" content="{SOCIAL_IMAGE_HEIGHT}">
  <meta property="og:image:alt" content="{safe_title}">
  <meta name="twitter:card" content="summary_large_image">
  <meta name="twitter:title" content="{safe_title}">
  <meta name="twitter:description" content="{safe_title}">
  <meta name="twitter:image" content="{safe_image}">
  <meta name="twitter:image:alt" content="{safe_title}">"""


def render_single_page(md_path: Path, html_path: Path, title: str, lang: str) -> None:
    article_html = render_markdown(md_path, lang)
    html_lang = "zh-CN" if lang == "zh" else "en"
    html_path.write_text(
        f"""<!doctype html>
<html lang="{html_lang}">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(title)}</title>
{social_meta(title)}
  <script>
{THEME_BOOTSTRAP}
  </script>
  <style>
{STYLE}
  </style>
</head>
<body>
  <main class="page">
    <article lang="{html_lang}">
{article_html}
    </article>
  </main>
  <script>
{COMMENTS_SCRIPT}
  </script>
</body>
</html>
"""
    )


def render_bilingual_page() -> None:
    en_html = render_markdown(EN_MD, "en")
    zh_html = render_markdown(ZH_MD, "zh")
    HTML_PATH.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{PAGE_TITLE_EN}</title>
{social_meta(PAGE_TITLE_EN)}
  <script>
{THEME_BOOTSTRAP}
  </script>
  <style>
{STYLE}
  </style>
</head>
<body>
  <main class="page">
    <div class="lang-switch-wrap" aria-label="Page controls">
      <div class="top-controls">
        <div class="lang-switch">
          <button type="button" id="lang-en" class="active" aria-pressed="true">English</button>
          <button type="button" id="lang-zh" aria-pressed="false">中文</button>
        </div>
        <button type="button" id="theme-toggle" class="theme-toggle" aria-pressed="false">Dark</button>
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
{LANG_SCRIPT}
{COMMENTS_SCRIPT}
  </script>
</body>
</html>
"""
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--single-md", type=Path)
    parser.add_argument("--single-html", type=Path)
    parser.add_argument("--title", default=PAGE_TITLE_EN)
    parser.add_argument("--lang", choices=["en", "zh"], default="zh")
    args = parser.parse_args()

    if args.single_md or args.single_html:
        if not args.single_md or not args.single_html:
            parser.error("--single-md and --single-html must be provided together")
        render_single_page(args.single_md, args.single_html, args.title, args.lang)
        return

    render_bilingual_page()


if __name__ == "__main__":
    main()
