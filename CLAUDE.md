## Active implementation plan

**Plan file:** `C:\Users\admin\.claude\plans\let-s-build-a-backtesting-valiant-orbit.md`
**Spec file:** `walk_forward_trading_brief_v2.md` (in this folder)

Before writing any code, read the plan. It defines the exact implementation order, file locations, function signatures, edge-case decisions, and verification steps. The plan is the source of truth for this build.

**Files completed so far:**
- `requirements.txt` ✅
- `filters.py` ✅
- `strategies/rsi_sma.py` ✅
- `strategies/__init__.py` ✅
- `backtester.py` ✅
- `components/__init__.py` ✅
- `components/charts.py` ✅
- `components/strategy_card.py` ✅
- `components/sidebar.py` ✅
- `app.py` ✅

**Build complete.** All files written and verified. Run with `streamlit run app.py`.

**Python 3.14 install note:** `pandas-ta` blocks Python 3.14 source builds. Workaround:
```
pip download pandas-ta --no-deps -d /tmp/pdta
pip install /tmp/pdta/pandas_ta-*.whl --no-deps
pip install tqdm  # undeclared dep of pandas-ta
```

---

## Critical rules
1. Don’t assume. Don’t hide confusion. Surface tradeoffs.

2. Minimum code that solves the problem. Nothing speculative.

3. Touch only what you must. Clean up only your own mess.

4. Define success criteria. Loop until verified.



**Superpowers skills** (invoke via `Skill` tool):
- `brainstorming` — explore intent and design before any implementation
- `writing-plans` — turn a spec into a step-by-step implementation plan
- `executing-plans` — run a written plan with review checkpoints
- `subagent-driven-development` — parallel agent execution of implementation tasks
- `dispatching-parallel-agents` — coordinate independent parallel tasks
- `systematic-debugging` — structured root-cause analysis before proposing fixes
- `test-driven-development` — red/green TDD workflow
- `verification-before-completion` — run verification before claiming work is done
- `requesting-code-review` — prepare and request a code review
- `receiving-code-review` — evaluate and respond to review feedback
- `finishing-a-development-branch` — decide how to integrate completed work
- `using-git-worktrees` — isolate feature work via git worktrees
- `writing-skills` — create or edit skill files
- `using-superpowers` — meta-skill: when and how to use all the above
- `grill-me` — stress-test a plan via relentless interviewing

**Command** (invoke via `/project:code-review <PR>`):
- `code-review` — multi-agent PR review with confidence-scored issues