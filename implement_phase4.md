# Kaggriculture - Phase 4 Implementation Plan (Validation & Advanced Logic)

This document outlines the design and implementation plan for Phase 4 of the Kaggriculture agent. Phase 4 focuses on advanced options (fertilizer buying, animal lifecycle placeholders) and the automated validation/release pipeline.

---

## 1. Goal
Complete the agent's full feature set. This includes implementing market fertilizer purchases, setting up placeholder routines for the animal lifecycle, and writing the scripts for validation, rule auditing, packaging, and Monte Carlo simulation.

---

## 2. Key Requirements & Actions to Implement

### A. Fertilizer Market Purchase
*   **Action Syntax**: Use market order `["BUY_PRODUCT", "FERTILIZER", qty]` (or simply `["BUY_PRODUCT", "FERTILIZER"]` if quantity is default) to purchase fertilizer.
*   **Rules constraint**: Bought fertilizer lands in the shed, meaning it counts against the 100-item shed capacity and must not be purchased if the shed is full.
*   **Application**: Once in the shed, workers can `PICKUP` fertilizer and apply it to crops using `CARE` to double the watering bonus for 3 days.

### B. Animal Lifecycle Gating (`ANIMALS_ENABLED`)
If `policy.ANIMALS_ENABLED` is flipped to `True` in `rules_validated.json`, the agent must support the full animal lifecycle:
1.  **Build Home**: Direct a worker to an empty tile and execute `["BUILD_COOP"]` (for geese) or `["BUILD_PASTURE"]` (for cows/sheep).
2.  **Buy Animal**: Issue a market order to buy the animal (e.g. `["BUY_PRODUCT", "GOOSE"]`). The animal arrives in the shed.
3.  **Transport**: A worker goes to the center shed, executes `["PICKUP", "GOOSE"]` (or the respective animal), carries it to the built home, and executes `["PLACE"]` standing on the home tile.
4.  **Feed Daily**: Every day, a worker must pick up 1 wheat from the shed, carry it to the animal home, and execute `["FEED"]`. If missed for 2 consecutive days, the animal escapes.
5.  **Care & Harvest**:
    *   Execute `["CARE"]` on the animal tile to bank a productivity bonus.
    *   Execute `["HARVEST"]` on the animal tile to collect animal products (eggs, milk, wool) once produced.

### C. Build & Release Scripts
1.  **`build_submission.py`**:
    *   Packages `main.py`, `scheduler.py`, `rules_validated.json`, and all helper modules into the final `submission.tar.gz` package.
    *   Must verify that the embedded rules inside the wrapper parse correctly and match the source file before creating the archive.
2.  **`validate_rules.py`**:
    *   Compares the local `rules_validated.json` file against the constants defined in the Kaggle environment engine (`kaggle_environments`) and reports any discrepancies.
3.  **`revalidate.py`**:
    *   Runs automated acceptance checks on the finished package (checking archive layout, latency constraints, and basic watering coverage).
4.  **`compliance_audit.py`**:
    *   Performs turn-by-turn checks of simulated seasons to detect illegal moves or wasted/wasted actions (such as watering an already fully watered plant).
5.  **`mc.py`**:
    *   Runs Monte Carlo matches across multiple seeds against starter and copy opponents to evaluate if new strategy configurations achieve the target release bar ($35,000).

---

## 3. Proposed File Changes

### A. Update: `scheduler.py`
*   Add logic to generate `BUY_PRODUCT` (fertilizer) tasks under appropriate conditions.
*   Flesh out the `if animals_on:` block in `generate_tasks` to produce animal build, placement, feeding, and care tasks.

### B. New Validation & Build Scripts
*   Create [`build_submission.py`](file:///c:/Users/91798/Desktop/Druidot_new/build_submission.py).
*   Create [`validate_rules.py`](file:///c:/Users/91798/Desktop/Druidot_new/validate_rules.py).
*   Create [`revalidate.py`](file:///c:/Users/91798/Desktop/Druidot_new/revalidate.py).
*   Create [`compliance_audit.py`](file:///c:/Users/91798/Desktop/Druidot_new/compliance_audit.py).
*   Create [`mc.py`](file:///c:/Users/91798/Desktop/Druidot_new/mc.py).

---

## 4. Verification Plan

### Automated Tests (`test_phase4.py`)
*   **Fertilizer Ordering**: Verify the scheduler generates `BUY_PRODUCT` orders for fertilizer when required.
*   **Animal Tasks**: Temporarily flip `ANIMALS_ENABLED` to `True` using pytest's `monkeypatch` and assert that the scheduler produces feed, care, and harvest tasks.
*   **Packaging Validation**: Run `build_submission.py` and verify that the output tarball is generated and validated.
