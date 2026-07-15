# Per-Module Reference

This page is a code-adjacent tour of each module in
`heuristic_learning/hl_benchmark/`. Every subsection lists the module's job,
its main functions, and the interface it exposes to the rest of the package.

## `envs.py`

**Job:** Own the fixed set of environments and seed splits, and describe the
runtime the ledger is running on.

**Key data:**

- `SEED_SPLITS` (`envs.py:12`) — the only place where the seed protocol
  (`dev = 0..19`, `holdout = 1000..1049`, `audit = 2000..2049`, `smoke = 0..1`)
  is defined.
- `ENV_SPECS` (`envs.py:36`) — a frozen `EnvSpec` per benchmark environment.
  Each spec carries the observation summary, action summary, reward
  interpretation, episode length, `success_target` (used as the strict pass
  cutoff), the description of the initial policy, and the known failure modes.
  This is the source of truth for the "Environments" section of every report.
- `OPTIONAL_SUBSTITUTIONS` (`envs.py:115`) — records the allowed fallback
  (`CarRacing-v3` for `BipedalWalker-v3`) when Box2D setup diverges.

**Key functions:**

- `get_seeds(split, seed_start=None, episodes=None)` at `envs.py:120` —
  either return the fixed seed range for `split`, or an explicit
  `[seed_start, seed_start + episodes)` range. Every seed used anywhere in the
  package flows through this helper.
- `benchmark_env_ids()` at `envs.py:135` — canonical iteration order.
- `spec_for(env_id)` at `envs.py:141` — lookup helper for reports.
- `make_env(env_id)` at `envs.py:162` and `check_env_available(env_id)` at
  `envs.py:169` — Gymnasium instantiation. `import_gymnasium()` raises with an
  actionable pip install message rather than letting an `ImportError` escape.
- `discover_runtime_metadata()` at `envs.py:180` — collects Python version,
  platform string, and installed versions of `gymnasium`, `box2d-py`, `Box2D`,
  `pygame`, `pygame-ce`, `swig`, `numpy`, `pandas`, `pytest`,
  `stable-baselines3`, and `torch`. Missing packages appear as
  `"not_installed"`.

## `policies/`

**Job:** Hold the transparent handwritten policies for each benchmark env.

The package is deliberately flat:

- `base.py` — `BasePolicy` interface (`reset`, `act`, `config`) and
  `RandomPolicy` baseline.
- `factory.py` — `make_policy(env_id, policy_name, action_space=None,
  config=None)` — dispatches on `(env_id, policy_name)` and folds a partial
  `config` dict into the env-specific `@dataclass` config using
  `_config_from_dict`.
- One module per environment:
  - `cartpole.py` — `CartPolePolicy` (linear sign controller with an optional
    structural "center-cart guard" mode).
  - `mountain_car.py` — `MountainCarPolicy` (energy pumping vs. a transparent
    finite-horizon planner that value-iterates on a precomputed grid).
  - `acrobot.py` — `AcrobotPolicy` (swing-up rule with an upright-region
    switch and a late-episode recovery kick) and `AcrobotDecisionTreePolicy`
    (an explicit sklearn-shaped tree distilled from a PPO teacher; loaded from
    `results/acrobot_tree_policy.json`).
  - `lunar_lander.py` — `LunarLanderPolicy` (PD controller with leg-contact
    and low-altitude guards).
  - `bipedal_walker.py` — `BipedalWalkerPolicy` (Fourier gait for the initial
    policy; three-mode `STAY_ON_ONE_LEG / PUT_OTHER_DOWN / PUSH_OFF` state
    machine for the improved policy).

**Policy naming convention** (`factory.py:38-40`):

- `random` — action-space sample; requires `action_space`.
- `initial` — env-specific config defaults, `structural=False`.
- `improved` — same config type, `structural=True`. Enables the extra guard,
  planner, or state machine in that env's policy class.
- `tuned` — same as `initial` shape-wise, but `config` overrides come from
  scalar search.
- `tree` (Acrobot only) — loads the teacher-distilled decision tree.

**Interface every policy honors** (`policies/base.py`):

```python
class BasePolicy:
    policy_name: str

    def reset(self, seed: int | None = None) -> None: ...
    def act(self, obs) -> action: ...
    def config(self) -> dict[str, Any]: ...
```

`config()` returns the dataclass fields plus a flag indicating whether the
structural mode is on. That dict is stored in the ledger `config` field, so a
policy version can be re-instantiated purely from the row.

## `evaluate.py`

**Job:** Run a policy on a fixed seed range and append one ledger row.

**Key functions:**

- `evaluate_policy(*, env_id, policy_name, split="dev", ...)` at
  `evaluate.py:49` — the primary entry. Handles env lifecycle,
  per-seed policy reset, per-episode rollout, `score_stats` aggregation, and
  error capture. On exception, it still writes a `pass_fail="fail"` row with
  the exception message in `failure_analysis`.
- `evaluate_many(...)` at `evaluate.py:136` — helper used by `--all` and by
  the CI `SPLIT` targets. Just iterates `(env_id, policy)` pairs.
- `main()` at `evaluate.py:178` — argparse dispatcher for the two modes
  (`--all` vs. single `--env`).
- `_load_config(config_json)` at `evaluate.py:27` — accepts either a literal
  JSON string or a filesystem path pointing to a JSON file.
- `_score_stats(scores)` at `evaluate.py:36` — computes
  `mean/std/median/min/max` (or all-`None` when no scores were collected).

**Contract with `ledger.py`:** every `evaluate_policy` call constructs a full
trial-entry dict via `make_trial_entry`, then calls `append_entry` and
`write_summary_csv`. There is no partial write path.

## `search.py`

**Job:** Explore a small scalar/config space on dev seeds, keep the best
config, and preserve every candidate as its own ledger row.

**Key functions:**

- `ensure_search_split_allowed(split)` at `search.py:19` — hard-fails on
  `"holdout"` and `"audit"`. This is the audit rule for scalar search.
- `candidate_configs(env_id, max_candidates=32)` at `search.py:26` — an
  `itertools.product` over a small, env-specific set of scalar knobs. The
  space is intentionally tiny (a few dozen candidates) so search cannot become
  a de-facto training loop.
- `run_search(*, env_id, ...)` at `search.py:104` — iterates candidates,
  calls `evaluate.evaluate_policy` with `policy_name="tuned"`,
  `change_type="scalar/config tuning"`, and tracks the best
  `mean - std_penalty * std`. Best config lands in
  `results/search_best_<env>.json`.

**Why it re-uses `evaluate_policy`:** the search harness deliberately owns no
seeding, rollout, or ledger logic of its own. That means a scalar search row
is indistinguishable from a manual `evaluate --policy tuned` row except for
the `change_summary` text.

## `rl_baseline.py`

**Job:** Optional deep-RL comparator. Only used if
`pip install -e '.[rl]'` was run.

**Key functions:**

- `load_hf_baseline(...)` at `rl_baseline.py:46` — download a pretrained
  Stable-Baselines3 model from Hugging Face Hub and rehydrate it with the
  local env's observation/action spaces.
- `train_ppo_baseline`, `train_dqn_baseline`, `train_sac_baseline` at
  `rl_baseline.py:78-215` — three algorithm-specific SB3 setups. All three
  keep `verbose=0`, `device="cpu"`, and use `MlpPolicy`.
- `evaluate_model(...)` at `rl_baseline.py:218` — the mirror image of
  `evaluate.evaluate_policy` for a trained SB3 model. Uses
  `deterministic=True` and the same fixed seed range.
- `train_evaluate_baseline(...)` at `rl_baseline.py:254` — the unified entry
  used by the CLI. Assembles the correct kwargs for the chosen algorithm,
  runs training (or loads pretrained), evaluates, and appends a ledger row
  with `change_type="rl/deep learning baseline"` and
  `environment_steps = eval_steps + train_steps` so training cost is visible.

**Naming rule:** pretrained rows use `policy_version="rl-<algo>-hf"`, trained
rows use `policy_version="rl-<algo>"`. Reports read this suffix to decide
whether a row is a locally trained baseline or a pretrained comparator.

## `ledger.py`

**Job:** Own the on-disk trial ledger and its derived CSV.

**Key functions:**

- `utc_now_iso()` at `ledger.py:46` — the only timestamp source used in this
  package.
- `git_commit()` / `diff_identifier()` at `ledger.py:59-73` — `git rev-parse`
  and `git diff` hashes. Both return `"unavailable"` (never crash) when git is
  missing or the tree is not a repo.
- `llm_cost_from_env()` at `ledger.py:76` — always writes the fixed
  `{calls, prompt_tokens, completion_tokens, total_tokens, source}` shape.
  The values are `"unavailable"` unless a future runtime plugs them in. This
  makes the reports' "LLM tokens: unavailable" line an honest signal, not a
  silent omission.
- `validate_entry(entry)` at `ledger.py:88` — checks the required-field set
  `REQUIRED_FIELDS` (defined at `ledger.py:21`). Every writer must produce a
  dict that passes this check.
- `append_entry(path, entry)` at `ledger.py:104` — the only place that opens
  `trials.jsonl`, and always in append mode.
- `read_entries(path)` at `ledger.py:113` — re-validates every line, so an
  invalid JSONL row raises with `path:line_number` context.
- `make_trial_entry(...)` at `ledger.py:133` — the factory the other modules
  use. Populates every required field, computes the `seed_range` from the
  seed list, and inserts `runtime_metadata` and `llm_cost` slots.
- `write_summary_csv(...)` at `ledger.py:202` — regenerates
  `results/summary.csv` from the ledger, with a fixed column order defined in
  the function body. The reports always call this before reading the CSV.

## `report.py`

**Job:** Turn the append-only ledger into a single stable Markdown report.

**Key functions:**

- `render_report(...)` at `report.py:446` — the entry point. Refreshes the
  summary CSV, reads all entries, deduplicates to a
  representative-per-`(env, policy, split)` view via `_representative_rows`,
  and writes `results/final_report.md`.
- Section builders: `_result_rows`, `_deep_rl_comparison_lines`,
  `_benchmark_threshold_lines`, `_audit_threshold_lines`,
  `_runtime_metadata_lines`, `_rl_baseline_status`,
  `_pretrained_comparator_lines`, `_cost_summary`, plus the summary strings
  `_holdout_conclusion`, `_deep_rl_goal_summary`,
  `_benchmark_threshold_summary`, `_audit_threshold_summary`.
- `_threshold_cutoff(success_target)` at `report.py:267` — the 5% tolerance
  rule (`0.95 * target` for positive targets, `target - 0.05 * |target|`
  for negative targets like MountainCar's `-110`).

**Rule of representative rows:** the report shows the latest entry per
`(env, policy_version, split)` triple, with special-cases for `rl-*` (keep
the best-mean run) and `tuned` on `dev` (keep the run with more episodes and
higher mean). Every raw entry is still in the ledger — the report just
collapses them for readability.

## `deepdive_report.py`

**Job:** A second report focused on the coding-agent process, not just the
scores.

**Key differences from `report.py`:**

- Tabulates per-env trial counts split by dev/holdout/audit, and by
  change-type (`_environment_process_rows` at `deepdive_report.py:192`).
- Lists non-pass iterations with their `failure_analysis` text
  (`_partial_trial_rows` at `deepdive_report.py:274`).
- Reports cost accounting (`_cost_lines` at `deepdive_report.py:292`)
  including "runs with episodes but zero recorded environment steps" (a
  bug-detector for logging/instrumentation errors).
- Runs `git status --short` for the benchmark path via
  `_git_status_for_project` at `deepdive_report.py:89`. `"clean"` in the
  report means the ledger was generated on a clean tree.

Both reports call `write_summary_csv` first so that the CSV cannot drift
away from the ledger.

## `summarize.py`

**Job:** A one-line CLI to regenerate `results/summary.csv` without touching
the reports. Only 22 lines; just calls `ledger.write_summary_csv`.

## `tests/`

Not part of the runtime, but they pin the schema in place. See `pytest tests`
via `make test`.

- `test_ledger_and_report.py` — validates required fields, append-only
  semantics, and report rendering against a synthetic ledger.
- `test_policies.py` — sanity checks that each policy returns valid actions
  and honors `structural=True/False`.
- `test_search.py` — checks that `ensure_search_split_allowed` refuses
  holdout/audit, and that `candidate_configs` is bounded.
- `test_evaluate_optional_gym.py` — guards the "Gymnasium not installed"
  error path.
- `test_rl_baseline.py` — guards the "Stable-Baselines3 not installed" error
  path.
