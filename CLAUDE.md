# Kaggriculture - Claude Code Project Context & Guidelines

This document provides a comprehensive guide for Claude Code regarding the Kaggriculture agent project: its purpose, rules, architecture, status, and remaining work.

---

## 1. Commands & Workflows
*   **Run Test Suite (Pytest)**: `python -m pytest test_phase2.py`
*   **Run Legacy Unit Tests**: `python test_phase1.py`
*   **Rule Validation**: (Future) `python validate_rules.py rules_validated.json`
*   **Build Submission Tarball**: (Future) `python build_submission.py`
*   **Audit Compliance**: (Future) `python compliance_audit.py`

---

## 2. Core Game Rules & Engine Contract (Non-Negotiable)
The Kaggle environment enforces strict, unforgiving execution limits. A single violation resets the season score to zero:
1.  **Format Constraints**: 
    *   Individual worker commands and market orders must be Python `list` structures, **never dictionaries** (e.g., `["WATER"]` is valid; `{"action": "water"}` is a silent no-op).
    *   The wrapper returns a dictionary of these lists: `{"farmer": [...], "hands": [...], "market": [...]}`.
2.  **Boots in the Dirt**: A worker must stand directly on the exact tile of a crop or weed to perform an action (`WATER`, `HARVEST`, `PLANT`, `DIG`). 
    *   *Shed Exception*: `DROP` and `PICKUP` work if standing on any of the four center squares: `(4,4)`, `(5,4)`, `(4,5)`, `(5,5)`.
3.  **Timing**: The agent must return within 1 second. `remainingOverageTime` starts at 60 seconds and does not refill. If it runs out, any timeout terminates the season.
4.  **No Monolithic Code**: The architecture is component-based. All strategies must be modular, passing state via clean data structures.
5.  **Market Limits**: Up to 10 market orders are processed per turn. Excess orders are ignored.
6.  **Price Floor**: All items hit a hard floor of $1 (price curves asymptotically decay towards it).
7.  **Season Length**: 720 turn steps (30 days of 24 hours). Terminal cash is the only metric that matters.

---

## 3. Current Project Architecture & Completed Components

The workspace is organized as a modular Python package:

```
├── rules_validated.json   # Game constants, crop rules, dials, priorities
├── rules_loader.py        # Loads and validates JSON rules (verifies 9 mandatory keys)
├── action_emitter.py      # Formats and wraps engine-compliant action outputs
├── telemetry.py           # In-memory logger tracking caught exception hour logs
├── market_model.py        # Stateless pricing math (linear, sq, sqrt, log) and crop EV ranking
├── care_monitor.py        # Dynamic capacity adjusting (shrinks on thirst, grows on idle crew)
├── forecaster.py          # Heuristic terminal bank projections & prediction calibration
├── main.py                # Public agent wrapper with full exception containment sandbox
├── test_phase1.py         # Standard unittest verification for Phase 1 components
└── test_phase2.py         # Pytest verification for Phase 2 components (using monkeypatch)
```

### Component Status (Phases 1 & 2 Completed)
*   **Wrapper Exception Containment**: Any unhandled exception is caught, logged in `telemetry`, and degrades to empty lists (`PASS`), preventing season termination.
*   **Rule Provenance**: Game rules reside only in `rules_validated.json`. `RulesLoader` verifies it before loading.
*   **Pricing Math**: `MarketModel` computes price decay based on stock and dynamically ranks crops via profit-per-turn EV.
*   **Monitoring**: `CareMonitor` adapts farm acreage capacity to match crew performance. `Forecaster` tracks realized earnings to adjust forecast calibration.

---

## 4. Remaining Roadmap

### Phase 3: The Operational Core (Scheduler & Routing) — *Next Up*
You need to implement `scheduler.py` to coordinate tasks and direct workers:
*   **Daily Tasks**: Issue `HIRE` commands up to `policy.target_hands` at hour 0. Sell shed inventory at hour 0 (max 10 orders).
*   **Watering Urgency**: Group tasks into 10 bands. WATER (9500) for crops with 1 missed watering must be priority #1 (2 misses kills the crop).
*   **Same-Day Planting Water**: A newly planted seed starts with 1 miss mark; it must be watered on the same day it is planted.
*   **Greedy Manhattan Matching**: Within each priority band, match workers to tasks by distance. Locked tiles are walkable. Workers step towards targets if not standing on them.
*   **DROP Pre-emption**: If carried load + shed capacity > `policy.drop_pressure` (80%), intercept the worker, remove them from scheduler bands, and direct them to the shed. Tighten threshold to 50% under overflow threat, and to 0 during endgame liquidation.

### Phase 4: Validation and Advanced Logic
*   **Animal Lifecycle**: Implemented and unit-tested for code-correctness (Build home -> Buy animal -> Pickup -> Place -> Feed daily -> Care/Harvest), but **gated behind `policy.ANIMALS_ENABLED`, which ships `false`**. The code path is exercised only by forcing the flag on in tests; it is **not** validated against the live Kaggle engine. See §5 before treating it as a usable capability.
*   **Fertilizer Logic**: Market fertilizer purchases are implemented but gated behind `policy.FERTILIZER_ENABLED` (ships `false`); same validation caveat as animals.
*   **Release Checks**: Create `build_submission.py` to check embedded rules match sources and package `main.py` + `rules_validated.json` into `submission.tar.gz`.
*   **Simulated Auditing**: Build `revalidate.py` (offline correctness) and `compliance_audit.py` (plays seasons to check for illegal/wasted actions).

---

## 5. Validation Status & Caveats (Non-Negotiable)

**Implemented ≠ validated.** Passing the unit suite proves code-level correctness
against our *own* model of the rules; it does **not** prove correctness against
the actual competition engine, nor competitive strength. Keep these distinct in
every status claim (register #31, #32, #35):

*   **Validated offline**: the engine contract (list-typed `{farmer, hands, market}`
    envelope, 10-order cap, $1 floor), exception containment + capture
    (`revalidate.py` REQ-01…REQ-07), rule-artifact parity (`validate_rules.py`,
    `build_submission.py`), and a clean compliance audit on the synthetic season.
*   **Implemented but gated OFF and engine-unvalidated**: the animal lifecycle
    (`ANIMALS_ENABLED`), market fertilizer (`FERTILIZER_ENABLED`), and the
    `town_model` demand helper (`demand_forecast`). All three flags ship `false`.
    Their code is unit-tested with the flag forced on for *correctness only* —
    do not describe them as working capabilities until demonstrated on the live
    engine.
*   **Not demonstrated here (needs `kaggle_environments`)**: exact-engine
    correctness and multi-opponent robustness (#31, #32); held-out / exogenous /
    overfitting evidence for the Bradley-Terry analysis (#29, #30, #33).
*   **`MIN_TERMINAL_TARGET` ($35,000) is a calibration baseline, not a universal
    release gate (#34)**: it was calibrated against idle/starter opponents on a
    limited setup. It is overridable (`mc.py --target`) and enforced only opt-in
    (`--require-target`). Recompute it when the opponent pool, season count, or
    evaluation variance changes.

The authoritative, per-item breakdown lives in `VALIDATION.md`.
