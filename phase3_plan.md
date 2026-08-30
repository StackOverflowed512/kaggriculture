# Kaggriculture - Phase 3 Implementation Plan (Scheduler & Routing)

This document outlines the design and implementation plan for Phase 3 of the Kaggriculture agent, focusing on the **Operational Core (Scheduler)** and worker pathfinding/routing.

---

## 1. Goal
Implement the `Scheduler` and worker task assignment system. The scheduler is responsible for scanning the map state, generating prioritized tasks, resolving worker movements via greedy Manhattan routing, and generating exact action commands.

---

## 2. Key Requirements & Rules to Implement

### A. Urgency-Based Task Priority Bands
Tasks must be grouped into 10 distinct priority bands (values read directly from `rules_validated.json["daily_routines"]` to avoid hardcoding):
1.  **Critical Watering (9500)**: Any crop at risk of dying (carrying 1 missed watering mark, going into the 2nd day without water).
2.  **Animal Harvesting (9100)**: Harvest ready animal products (inactive while `ANIMALS_ENABLED` is false).
3.  **Crop Harvesting (9000 + crop value)**: Pick ripe one-time crops (wait for the full window closes) or repeaters.
4.  **Animal Feeding (8800)**: Feed animals wheat from hands (inactive while `ANIMALS_ENABLED` is false).
5.  **Bonus Watering (8000)**: Water one-time crops inside their yield-bonus window.
6.  **Ongoing Watering (7000)**: Water repeaters (tomato/strawberry) to ensure survival.
7.  **Crop Planting (6000)**: Plant new crops *and* schedule immediate watering on the same day.
8.  **Weed Clearing (5500)**: Dig out weeds.
9.  **Normal Watering (5000)**: Water crops with nothing else urgent.
10. **Animal Care/Fertilizer (3500-4000)**: Collect fertilizer or care for animals (inactive while `ANIMALS_ENABLED` is false).

### B. Special Constraints
*   **Same-Day Planting Water**: A newly planted seed starts with 1 missed watering mark immediately. If not watered on the turn/day it is planted, it gets a 2nd mark at midnight and turns into a weed. Planting must trigger an immediate same-day watering task.
*   **Acreage Limits**: Respect the dynamic workable acreage limit calculated by `CareMonitor`. Do not plant outside the active capacity quadrant boundary.
*   **The DROP Pre-emption**:
    *   Triggered when a worker's carried load + current shed usage > `policy.drop_pressure` (80% / 80 items by default).
    *   The threshold must tighten to half the shed under overflow risk (e.g. at end of day) and to 0 during endgame liquidation.
    *   Triggered workers bypass the scheduler bands, are assigned a direct path to the nearest shed tile (4,4), (5,4), (4,5), (5,5), and issue a `DROP` command once adjacent.

### C. Distance-Aware Worker Matching (Greedy Manhattan)
*   For each band, workers are matched to tasks using Manhattan distance.
*   The board contains no obstacles, and locked tiles are walkable.
*   If a worker is not standing directly on the task tile, they must issue a move action (`["NORTH"]`, `["SOUTH"]`, `["EAST"]`, or `["WEST"]`) towards it. 
*   If standing directly on the tile, they execute the work action (`["WATER"]`, `["HARVEST"]`, etc.).
*   Exception: Shed drop-off/pick-up works from any of the four center squares: (4,4), (5,4), (4,5), (5,5).

### D. Daily Routine Actions (Hour 0)
*   **Hiring**: At hour 0 of every day (where `step % 24 == 0`), issue `HIRE` commands up to `policy.target_hands`.
*   **Selling**: At hour 0 of every day during normal operations, phone the commodity broker to sell what is currently in the shed (capped at 10 orders per turn).

### E. Endgame Overlay Rules (Turn 670+)
When the step reaches `policy.endgame_start_turn` (default 670), the scheduler activates the endgame overlay:
1.  **Stop Land & Animal Buying**: Disable all logic for purchasing new land deeds or animals.
2.  **Liquidation Sell Actions**: Phone the broker **every turn** (not just hour 0) to sell all available shed inventory, up to the 10-call hourly limit.
3.  **Forced Drops**: Drop threshold (`drop_pressure`) is set to 0. Any worker carrying any harvestable item is immediately redirected to drop it at the shed.
4.  **Planting Cash Test**: Check crop-by-crop every morning if a crop can be fully harvested and sold before turn 720. If not, stop planting it. If no crops pass, stop planting completely.


---

## 3. Proposed Component Changes

### A. New File: `scheduler.py`
Implement `Scheduler` class with:
- `generate_tasks(obs, rules, care_capacity)`: Scans the 10x10 grid and generates a list of task objects (containing type, coordinates, priority, and target crop type).
- `assign_tasks(workers, tasks, rules)`: Performs the band-by-band greedy matching of workers to tasks, resolving conflicts and generating step-by-step actions.

### B. Update: `main.py`
- Integrate `Scheduler` into the main execution cycle.
- Manage persistent state of workers (positions, carried loads).
- Call `Scheduler` to get farmer and hired-hand actions.
- Issue daily `HIRE` and `SELL` calls at hour 0.

---

## 4. Verification Plan

### Automated Tests (`test_phase3.py`)
- **Priority Sorting**: Verify task list properly sorts dying waterings (9500) above standard harvesting or planting.
- **Same-Day Watering**: Verify planting a seed generates an immediate watering task for that cell.
- **Manhattan Routing**: Mock a worker at (0,0) and a task at (2,3) and verify they generate step-by-step direction commands toward the target.
- **DROP Threshold**: Verify a worker carrying items above the drop threshold is intercepted, ignores scheduler bands, and heads straight to the shed.
