# AgentArena

Grades LLM agents on **tool use** and **research** across multiple metrics, with an arena-style
leaderboard. All models run through the same agent loop, prompts, tools and budgets.

## Results

Full suite: 16 task templates × 3 seeds × 3 repeats = 144 samples per model (October 2026).

| Rank | Model | Elo (95% CI) | pass@1 | Tool use | Research | pass^3 | Cost / solved |
|---|---|---|---|---|---|---|---|
| 1 | glm-5.3 | 1010 (1005–1015) | 100% | 100% | 100% | 100% | $0.0071 |
| 1 | claude-sonnet-5-5 | 1007 (1002–1013) | 100% | 100% | 100% | 100% | $0.0128 |
| 3 | glm-4.7-flash | 983 (975–990) | 94% | 89% | 100% | 85% | free |

- glm-5.3 and Sonnet 5.5 are statistically tied. Both solved every sample, and glm-5.3 did it
  at about half the cost. The whole run cost $2.87.
- glm-4.7-flash matched them on research but failed 8 of 72 tool-use samples. It cancelled the
  oldest order instead of the most recent, read an email without replying, and bought the wrong
  item. Its pass^3 of 85% shows these failures were intermittent.
- **The suite is now saturated for frontier models.** Harder templates are needed to separate them.

## What it measures

| | |
|---|---|
| Success | pass@1 with clustered-bootstrap CIs; **pass^k** (all k attempts succeed) for reliability |
| Ranking | Bradley-Terry ratings on the Elo scale; ties when CIs overlap |
| Efficiency | cost per solved task, tokens, steps, latency |
| Process | invalid / redundant tool calls, loops, error recovery |
| Research | correctness plus citations verified verbatim against the source docs |

Design choices:
- Tool-use tasks are graded on the **final world state**, compared against a replayed gold
  solution (tau-bench style).
- Research runs over a **fictional corpus**, so models can't answer from memory.
- Each template contains **traps**: sold-out items, look-alike names, outdated facts, and
  unanswerable or infeasible requests.
- **API errors are excluded from scores** and retried on resume.

## Usage

```
uv sync
cp .env.example .env                                      # ANTHROPIC_API_KEY, ZAI_API_KEY
uv run arena run -m glm-5.3,claude-sonnet-5-5 -s smoke    # or: edit and run run.py
uv run arena report <run_id> --open
```

Models and prices live in `config/models.yaml`, and suites in `config/suites/`. The `fake-*`
models run the whole pipeline offline at no cost. `uv run arena --help` lists the other
commands: rescore, judge, validate-judge and runs.
