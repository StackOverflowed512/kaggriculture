# Kaggriculture — Validation Status

This document records **what has actually been demonstrated** versus **what is
merely implemented**, so no reader mistakes a green unit suite for proof of
competition readiness. It is the authoritative companion to the bug & defect
register (`kaggriculture_bug_report.pdf`) and is referenced from `CLAUDE.md` §5.

> **One-line summary:** every engine-*independent* defect in the register is
> fixed and covered by an offline test or release check. Everything that needs
> the real `kaggle_environments` engine — exact-engine correctness, live
> multi-opponent results, and the live animal economy — is **explicitly
> deferred, not claimed.**

---

## 1. Validation tiers

| Tier | Meaning | How it is demonstrated here |
|------|---------|-----------------------------|
| **A — Validated offline** | Proven against our model of the rules, with an automated check that fails loudly on regression. | `pytest`, `revalidate.py`, `validate_rules.py`, `build_submission.py --check-only`, `compliance_audit.py` |
| **B — Implemented, gated OFF, engine-unvalidated** | Code exists and is unit-tested *for correctness only* with its flag forced on; the shipped rules keep the flag `false`, so it never runs in production and has never been measured on the live engine. | Flag-forced unit tests (`_with_policy(rules, FLAG=True)`) |
| **C — Not demonstrated (needs the engine)** | Requires `kaggle_environments`, which is not installed in this workspace. Harnesses are wired and unit-tested; only the live *numbers* are missing. | Deferred — documented, not fabricated |

**Tier B and Tier C claims must never be reported as working capabilities.**
This is the direct mitigation for register item **#35** (documentation must not
exceed measured reliability).

---

## 2. Tier A — validated offline

* **Engine output contract** — every turn returns the `{farmer, hands, market}`
  envelope with list-typed actions and ≤ `max_market_orders` orders
  (`revalidate.py` REQ-01; `action_emitter` tests).
* **Exception containment** — a malformed observation degrades to `PASS` and is
  counted, never raised (REQ-02). A well-formed season raises **zero** contained
  exceptions (REQ-06); a contained exception is **fully captured** with
  step/type/category/traceback and the by-category aggregate reconciles to the
  count (REQ-07, ties to register #16/#17).
* **Latency** — worst synthetic turn is well under the 1 s budget (REQ-03).
* **Watering coverage** — workers standing on about-to-die crops all `WATER`
  them (REQ-04).
* **Rule-artifact provenance** — `rules_validated.json` and the embedded copy in
  `rules_loader.py` stay semantically identical; `validate_rules.py` checks the
  invariants and `build_submission.py` regenerates the single-file build fresh to
  verify parity (register #18).
* **Compliance audit** — a synthetic season produces no illegal or wasted
  actions (`compliance_audit.py`).

Run the full Tier-A gate:

```bash
python -m pytest
python revalidate.py
python compliance_audit.py
python validate_rules.py rules_validated.json
python build_submission.py --check-only
```

---

## 3. Tier B — implemented, gated OFF, engine-unvalidated

| Capability | Flag (ships `false`) | Correctness tests | Register items |
|------------|----------------------|-------------------|----------------|
| Animal lifecycle (build → buy → pickup → place → feed → harvest/care) | `ANIMALS_ENABLED` | `test_manager_bugs.py` (#2, #6, #7, #8, #9, #10, #13, #14), `test_audit_fixes.py` (#13) | #2, #6, #7, #8, #9, #13 |
| Market fertilizer purchases | `FERTILIZER_ENABLED` | fertilizer-path unit tests | #11 |
| Town demand approximation | `demand_forecast` | `test_manager_bugs.py` (#19) | #19 |

**What "unit-tested for correctness only" means:** the tests force the flag on to
prove the *code* does the right thing (e.g. feed is rationed to available wheat,
a carried animal is locked to its placement tile, demand is sourced entirely from
`town_model`). They do **not** establish that the behaviour is profitable or even
legal on the live engine. Until that is demonstrated, these remain **off**.

Register **#1** ("animal economy not end-to-end enabled") and **#3** ("wheat/
animal bootstrap") are intentionally **left gated**: enabling the live animal
economy is an engine-validated decision, out of scope for an offline workspace.

---

## 4. Tier C — not demonstrated (needs `kaggle_environments`)

These are **not bugs in our code**; they are evidence that can only be produced by
running the real engine, which is unavailable here. The harnesses exist and are
unit-tested offline (`mc.py`, `bt_model.py`); only the live numbers are pending.

* **#31 — exact-engine correctness**: the unit suite tests our model of the
  rules, not the competition engine's complete action/state semantics.
* **#32 — exact-engine multi-opponent validation**: no live head-to-head results.
* **#29 — held-out BT validation**, **#30 — exogenous opponent pool**,
  **#33 — overfitting/underfitting evidence**: the Bradley-Terry analysis
  (`bt_model.py`) is implemented and unit-tested (Wilson intervals, disconnected-
  graph handling, zero-win regularization — register #25/#26/#27/#28), but the
  generalization *evidence* requires live matches.

To produce Tier-C evidence once the engine is available:

```bash
python mc.py --matches 50 --require-engine          # live terminal-cash spread
python mc.py --tournament --require-engine           # Bradley-Terry over the pool
```

---

## 5. The $35,000 threshold (#34)

`MIN_TERMINAL_TARGET` ($35,000 in `rules_validated.json`) is a **calibration
baseline**, measured against idle/starter opponents on a limited reference setup.
It is **not** a universal competition-readiness gate:

* It is **overridable** — `python mc.py --target <value>`.
* Its enforcement is **opt-in** — only `--require-target` turns a below-target
  mean into a non-zero exit.
* A different matchup is a different measurement. Self-play, for instance, has two
  identical farms splitting one market; its worst case was never what this bar was
  meant to clear.

Recompute the bar whenever the opponent pool, season count, or evaluation
variance changes. `validate_rules.py` only enforces the internal invariant
`MIN_TERMINAL_TARGET <= CORRECTION_THRESHOLD`, not the bar's external validity.

---

## 6. Register disposition at a glance

* **Fixed (engine-independent), Tier A or B**: #2, #4, #5, #6, #7, #8, #9, #10,
  #11, #12, #13, #14, #15, #16, #17, #18, #19, #20, #21, #22, #23, #24, #25, #26,
  #27, #28.
* **Documentation fixes**: #34 (this file + `mc.py`/`CLAUDE.md`), #35 (this file
  + `CLAUDE.md` §5).
* **Deferred to the engine (Tier C / gated)**: #1, #3 (live animal economy —
  gated), #29, #30, #31, #32, #33 (live-engine evidence).
