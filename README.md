# Kaggriculture Agent

Kaggriculture is an autonomous, component-based farming agent designed to run in the Kaggle environment. The agent operates a farm over a fixed 30-day (720-hour) season, aiming to maximize terminal cash in the register at the end of the season while complying with strict execution constraints.

---

## Project Overview

The Kaggriculture agent coordinates a farmer (manager) and up to 10 hired day-laborers on a 10x10 grid. It manages the full lifecycle of crops (planting, watering, weeding, harvesting) and handles inventory transactions via a commodity broker. 

### Non-Negotiable Engine Constraints
*   **Command Formats**: All commands for workers (steps, planting, watering, weeding, harvesting) and market orders (hiring, buying seeds, selling) must be Python `list` structures, not dictionaries.
*   **Boots in the Dirt**: Workers must stand directly on a tile to interact with it. The only exception is the central shed tiles `(4,4)`, `(5,4)`, `(4,5)`, `(5,5)`, which allow dropping off or picking up items from adjacent center tiles.
*   **Timeout & Overage**: Every turn must execute within 1 second. Occasional latency spikes are covered by a non-refilling 60-second piggy bank (`remainingOverageTime`).
*   **Exception Containment**: A single unhandled exception or illegal action format zeroes the score for the entire season. The agent entry point must never crash.
*   **Market Limit**: At most 10 market actions (buying, selling, hiring) are processed per turn.
*   **Price Floor**: All items have a hard price floor of $1.

---

## Current Architecture & Implementation Status

The project is built as a highly decoupled, modular system of components that pass state via well-defined structures, avoiding a monolithic design.

### Implemented Components

1.  **Wrapper & Exception Containment (`main.py`)**: 
    *   Exposes the Kaggle entry point `agent(obs, config)`.
    *   Wraps execution in a sandbox that catches any unexpected runtime error, logs it in telemetry, and degrades to a safe `PASS` action to preserve the season.
    *   Tracks worker positions, daily hiring, and morning sales.
2.  **Rules Loader (`rules_loader.py` & `rules_validated.json`)**: 
    *   Validates and parses the external configuration rulebook containing game laws, dials, priorities, and constants. 
    *   Verifies 12 mandatory constants before loading, reverting to embedded defaults if the external file is corrupted or missing keys.
3.  **Action Emitter (`action_emitter.py`)**: 
    *   Enforces the list structure for all commands and truncates market orders to the 10-call hourly limit.
4.  **Telemetry (`telemetry.py`)**: 
    *   Tracks in-memory exception metrics and the step hours when they occurred.
5.  **Market Model (`market_model.py`)**: 
    *   Stateless pricing functions that calculate logarithmic, linear, square, and square-root price curves.
    *   Computes the dynamic Economic Value (EV) per tile per turn for crop ranking.
6.  **Care Monitor (`care_monitor.py`)**: 
    *   Statefully tracks watering coverage and shrinks target acreage capacity when plants miss water (thirst), slowly expanding it when the crew is idle.
7.  **Forecaster (`forecaster.py`)**: 
    *   Projects terminal cash based on current cash, shed value, and expected crop yields. Updates a calibration ratio by comparing predictions with actual outcomes.
8.  **Scheduler (`scheduler.py`)**: 
    *   **Priority Bands**: Generates tasks matching 10 urgency priorities (e.g., critical watering, harvesting, planting, weeding) defined in the configuration.
    *   **Same-Day Planting Water**: Ensures newly planted seeds are watered on the same turn to avoid turning into weeds at midnight.
    *   **Greedy Manhattan Matching**: Matches workers to tasks within each priority band by closest distance.
    *   **DROP Pre-emption**: Intercepts workers carrying goods if they exceed a dynamic drop threshold (80% normally, 50% under overflow threat, 0 during endgame) and directs them straight to the shed.
    *   **Endgame Overlay**: Triggered at turn 670. Disables purchases, forces all workers to drop carried goods, uses a cash-test cutoff to stop planting crops that won't mature in time, and liquidates all shed items by calling the broker every turn.

---

## Testing & Verification

A robust suite of tests verifies the correctness of each component:
*   `test_phase1.py` & `test_phase2.py`: Verify the foundations, rules loader, action emitter, and pricing model.
*   `test_phase3.py`: Verification suite (27 pytest assertions) covering scheduler task priorities, Manhattan step routing, same-day watering, drop pre-emption, capacity controls, and the endgame overlay.

To run the tests:
```bash
python -m pytest test_phase3.py
```
