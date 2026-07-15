# Framework Call Flow

This page shows what actually happens when you run one CLI command. Each
diagram is a sequence trace through the module graph in
[`architecture.md`](architecture.md).

## `make eval-env ENV=CartPole-v1 POLICY=initial SPLIT=dev`

Evaluate one policy on one environment over one seed split, and append one
ledger row.

```mermaid
sequenceDiagram
    autonumber
    actor CLI as "make eval-env"
    participant Ev as "evaluate.py"
    participant En as "envs.py"
    participant Pf as "policies.factory"
    participant Pol as "CartPolePolicy"
    participant Env as "Gymnasium env"
    participant Lg as "ledger.py"
    participant Meta as "envs.discover_runtime_metadata"
    participant FS as "filesystem<br/>(results/)"

    CLI->>Ev: python -m hl_benchmark.evaluate<br/>--env CartPole-v1 --policy initial --split dev
    Ev->>Ev: main() parses argparse
    Ev->>Ev: evaluate_policy(env_id, policy_name, split=dev)
    Ev->>En: get_seeds("dev") -> [0..19]
    Ev->>En: make_env("CartPole-v1")
    En->>Env: gym.make(env_id)
    Ev->>Pf: make_policy("CartPole-v1", "initial", action_space)
    Pf->>Pol: CartPolePolicy(CartPoleConfig(), structural=False)
    loop for each seed in seeds
        Ev->>Env: env.reset(seed=seed)
        Ev->>Pol: policy.reset(seed)
        loop until terminated or truncated
            Ev->>Pol: action = policy.act(obs)
            Ev->>Env: obs, reward, terminated, truncated, _ = env.step(action)
        end
        Ev->>Ev: append per-episode {seed, score, steps}
    end
    Ev->>Env: env.close()
    Ev->>Lg: make_trial_entry(env_id, policy_version, config, ..., score_stats)
    Lg->>Lg: utc_now_iso(), git_commit(), diff_identifier()
    Lg->>Meta: discover_runtime_metadata()
    Meta-->>Lg: {python, platform, packages}
    Lg-->>Ev: full trial-entry dict
    Ev->>Lg: append_entry(results/trials.jsonl, entry)
    Lg->>Lg: validate_entry(entry)
    Lg->>FS: open(trials.jsonl, "a").write(json.dumps(entry) + "\n")
    Ev->>Lg: write_summary_csv(trials.jsonl, summary.csv)
    Lg->>FS: read all entries, rewrite summary.csv
    Ev-->>CLI: print("pass CartPole-v1 initial mean=... steps=...")
```

Key invariants visible in the trace:

- The env is instantiated exactly once and closed in a `finally` block
  (`evaluate.py:78-101`); a policy exception does not leak an env handle.
- Every row is validated by `validate_entry` before it hits disk. A missing
  required field raises `ValueError` and the row is not appended.
- `write_summary_csv` re-reads the entire JSONL, so the CSV always matches the
  ledger.

## `make search MAX_CANDIDATES=32`

Run scalar/config search on the dev split for every registered env, one after
the other. Each candidate becomes one ledger row.

```mermaid
sequenceDiagram
    autonumber
    actor CLI as "make search"
    participant Sr as "search.py"
    participant En as "envs.py"
    participant Ev as "evaluate.py"
    participant Lg as "ledger.py"
    participant FS as "filesystem"

    CLI->>Sr: python -m hl_benchmark.search --all --split dev
    Sr->>Sr: main() parses argparse
    Sr->>Sr: ensure_search_split_allowed("dev") — refuses holdout/audit
    Sr->>En: benchmark_env_ids() -> [CartPole, MountainCar, ...]
    loop for each env_id
        Sr->>Sr: run_search(env_id, split="dev")
        Sr->>Sr: candidate_configs(env_id, max_candidates)
        Note over Sr: env-specific product of scalar knobs<br/>e.g. CartPole = pole/cart gains
        loop for each config in candidates
            Sr->>Ev: evaluate_policy(env_id, "tuned",<br/>split="dev", config=config)
            Ev->>Lg: make_trial_entry(...) with change_type="scalar/config tuning"
            Ev->>Lg: append_entry(results/trials.jsonl, entry)
            Ev-->>Sr: entry
            Sr->>Sr: score = mean - std_penalty * std<br/>track best
        end
        Sr->>FS: write results/search_best_<env>.json (best config)
    end
    Sr-->>CLI: print "best CartPole: mean=... score=... config=..."
```

Why `search.py` is separate from `evaluate.py`:

- `search.py:19` (`ensure_search_split_allowed`) hard-fails on holdout/audit.
  Structural evaluation runs freely on any split, but scalar tuning is
  segregated so it cannot silently overfit to reserved seeds.
- Every candidate goes through `evaluate.evaluate_policy` unchanged. Scalar
  search is "many small evaluations" rather than a separate code path.

## `make report`

Rebuild `results/final_report.md` from the append-only ledger.

```mermaid
sequenceDiagram
    autonumber
    actor CLI as "make report"
    participant Rp as "report.py"
    participant Lg as "ledger.py"
    participant En as "envs.py"
    participant FS as "filesystem"

    CLI->>Rp: python -m hl_benchmark.report
    Rp->>Rp: render_report()
    Rp->>Lg: write_summary_csv(trials.jsonl, summary.csv)
    Lg->>FS: rewrite summary.csv from ledger
    Rp->>Lg: read_entries(trials.jsonl)
    Lg-->>Rp: [entry, ...]
    Rp->>Rp: _read_summary(summary.csv)
    Rp->>Rp: _representative_rows(rows)<br/>latest-per-(env, policy, split)
    Rp->>En: iterate ENV_SPECS for env sections
    Rp->>Rp: _rl_baseline_status(entries)
    Rp->>Rp: _deep_rl_comparison_lines(rows)
    Rp->>Rp: _benchmark_threshold_lines(rows)
    Rp->>Rp: _audit_threshold_lines(rows)
    Rp->>Rp: _holdout_conclusion(rows)
    Rp->>FS: write results/final_report.md
    Rp-->>CLI: print("wrote results/final_report.md")
```

The deep-dive report (`make deepdive-report`) follows the same shape but calls
`deepdive_report.render_deepdive_report()` instead, and adds per-env process
timelines and non-pass iteration tables.

## `make rl-baseline ENV=CartPole-v1 ALGO=ppo TRAIN_STEPS=10000`

Optional PPO/DQN/SAC comparator. This is an entirely separate CLI; it does not
modify the transparent-policy pipeline. Its ledger rows land in the same
`trials.jsonl` but with `policy_version="rl-ppo"` (or `rl-dqn` / `rl-sac` /
`rl-sac-hf`) and `change_type="rl/deep learning baseline"`.

```mermaid
sequenceDiagram
    autonumber
    actor CLI as "make rl-baseline"
    participant Rl as "rl_baseline.py"
    participant En as "envs.py"
    participant Sb3 as "Stable-Baselines3"
    participant Env as "Gymnasium env"
    participant Lg as "ledger.py"
    participant FS as "filesystem"

    CLI->>Rl: python -m hl_benchmark.rl_baseline --env X --algo ppo --train-steps N
    Rl->>Rl: train_evaluate_baseline(algorithm="ppo", ...)
    alt pretrained_repo_id is set
        Rl->>Sb3: load_hf_baseline() via huggingface_hub
    else train from scratch
        Rl->>En: make_env(env_id)
        Rl->>Sb3: PPO("MlpPolicy", env, ...).learn(train_steps)
    end
    Rl->>Rl: evaluate_model(model, env_id, split=dev/holdout)
    loop for each seed in split
        Rl->>Env: env.reset(seed=seed)
        Rl->>Sb3: model.predict(obs, deterministic=True)
        Rl->>Env: env.step(action)
    end
    Rl->>Lg: make_trial_entry(policy_version="rl-ppo",<br/>change_type="rl/deep learning baseline")
    Rl->>Lg: append_entry(trials.jsonl, entry)
    Rl->>Lg: write_summary_csv(trials.jsonl, summary.csv)
    Rl-->>CLI: print("pass CartPole-v1 rl-ppo train_steps=N mean=...")
```

Note: `environment_steps` in the RL row is `eval_steps + train_steps`, so the
cost of training is visible in the same column that the heuristic rows use for
env interaction.
