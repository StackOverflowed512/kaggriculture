"""scheduler.py -- Phase 3 Operational Core (Scheduler & Routing).

The Scheduler is the operational heart of the agent. Each turn it:
    1. Scans the board state and builds a list of prioritized ``Task`` objects
       grouped into the ten urgency bands (Section 2.A of the plan).
    2. Pre-empts any worker whose load would threaten the shed and sends them to
       drop (Section 2.B / the DROP pre-emption).
    3. Matches the remaining workers to tasks band-by-band using greedy Manhattan
       distance and emits the exact engine action for each worker.

Provenance: every band priority is read from ``rules_validated.json``
(``daily_routines``) rather than hard-coded, so tuning happens in data only.

Coordinate convention (documented and self-consistent):
    A position is an ``(x, y)`` tuple, 0-indexed on the 10x10 board.
        x = column  -> EAST increases x, WEST decreases x
        y = row     -> SOUTH increases y, NORTH decreases y
    The NW quadrant is the top-left (small x, small y), matching the land-deed
    layout in the handover. The shed's four centre tiles are (4,4), (5,4),
    (4,5), (5,5); DROP/PICKUP are the only actions that work while merely
    standing on one of those tiles (every other action is "boots in the dirt").
    LOCKED tiles are walkable, so routing needs no obstacle avoidance.
"""

from market_model import MarketModel

# The four centre tiles that give access to the shed (DROP / PICKUP).
SHED_TILES = ((4, 4), (5, 4), (4, 5), (5, 5))


def manhattan(a, b):
    """Manhattan (taxicab) distance between two ``(x, y)`` positions."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def step_towards(pos, target):
    """Return a single one-step move action list toward ``target``.

    Movement is relative and one step per turn (engine contract). We close the
    x-axis first, then the y-axis; the exact order is arbitrary on an
    obstacle-free board as long as each step strictly reduces the distance.
    Returns ``[]`` when the worker already stands on the target tile.
    """
    px, py = pos
    tx, ty = target
    if px != tx:
        return ["EAST"] if tx > px else ["WEST"]
    if py != ty:
        return ["SOUTH"] if ty > py else ["NORTH"]
    return []


def nearest_shed_tile(pos):
    """Return the closest shed centre tile to ``pos`` (deterministic on ties)."""
    return min(SHED_TILES, key=lambda tile: manhattan(pos, tile))


class Task:
    """A single unit of work the scheduler wants done at a board tile.

    Attributes:
        kind:     one of WATER / HARVEST / PLANT / DIG / FEED / CARE / DROP, or
                  an animal-setup verb (BUILD_COOP / BUILD_PASTURE / PLACE).
        pos:      the ``(x, y)`` tile the worker must stand on to act.
        priority: the urgency band value (higher runs first).
        value:    in-band tie-breaker; used for HARVEST as the "+ crop value"
                  term so richer crops are picked first without crossing bands.
        crop:     crop type -- the argument for a PLANT action, or context.
        same_day: True for the watering task paired with a fresh PLANT (a seed
                  starts one miss from death and must be watered the same day).
        item:     argument for actions that take one (PICKUP <item>); also the
                  animal type carried for a PLACE.
        fetch:    an item that must be picked up from the shed *before* the
                  worker can perform ``kind`` at ``pos`` (FEED needs "WHEAT", a
                  PLACE needs the animal). None means "act directly on arrival".
    """

    __slots__ = ("kind", "pos", "priority", "value", "crop", "same_day", "item", "fetch")

    def __init__(self, kind, pos, priority, value=0.0, crop=None, same_day=False,
                 item=None, fetch=None):
        self.kind = kind
        self.pos = tuple(pos)
        self.priority = priority
        self.value = value
        self.crop = crop
        self.same_day = same_day
        self.item = item
        self.fetch = fetch

    def action(self):
        """The engine action list to emit when a worker stands on ``pos``."""
        if self.kind == "PLANT":
            return ["PLANT", self.crop]
        if self.kind == "PICKUP":
            return ["PICKUP", self.item]
        return [self.kind]

    def sort_key(self):
        """Descending sort key: band first, then in-band crop value."""
        return (self.priority, self.value)

    def __repr__(self):
        return (
            f"Task({self.kind}, pos={self.pos}, pri={self.priority}, "
            f"val={self.value}, crop={self.crop}, same_day={self.same_day}, "
            f"item={self.item}, fetch={self.fetch})"
        )

    def __eq__(self, other):
        if not isinstance(other, Task):
            return NotImplemented
        return (
            self.kind == other.kind
            and self.pos == other.pos
            and self.priority == other.priority
            and self.value == other.value
            and self.crop == other.crop
            and self.same_day == other.same_day
            and self.item == other.item
            and self.fetch == other.fetch
        )


class Scheduler:
    """Builds prioritized tasks and routes workers via greedy Manhattan matching."""

    def __init__(self):
        # Drop mode governs the DROP pre-emption threshold. main.py sets it each
        # turn: "normal" (80%), "overflow" (half shed, e.g. end of day), or
        # "endgame" (0 -- liquidate every load).
        self.drop_mode = "normal"

    # ------------------------------------------------------------------ #
    # Task generation
    # ------------------------------------------------------------------ #
    def generate_tasks(self, obs, rules, care_capacity, market_stocks=None):
        """Scan the board and return a list of prioritized ``Task`` objects.

        ``obs`` is a plain dict describing the board this turn. Recognised keys
        (all optional -- absent means "nothing of that kind"):
            step:          current turn index (for the planting cash test).
            crops:         list of crop dicts, each with at least ``pos``. Other
                           fields: ``type``, ``misses`` (missed waterings),
                           ``ready`` (ripe to harvest), ``needs_water``,
                           ``in_bonus_window``, ``repeater``.
            weeds:         list of ``(x, y)`` tiles to dig out.
            empty_tiles:   list of plantable ``(x, y)`` tiles.
            market_stocks: dict of item -> current market inventory.
        """
        dr = rules["daily_routines"]
        policy = rules.get("policy", {})
        animals_on = policy.get("ANIMALS_ENABLED", False)
        to_die = rules["constants"]["missed_waterings_to_die"]
        step = obs.get("step", 0)

        crops = obs.get("crops") or []
        weeds = obs.get("weeds") or []
        empty_tiles = obs.get("empty_tiles") or []
        if market_stocks is None:
            market_stocks = obs.get("market_stocks") or {}

        tasks = []

        # --- Standing crops: harvest when ripe, water by urgency band --------
        for crop in crops:
            pos = tuple(crop["pos"])
            ctype = crop.get("type")
            is_repeater = self._is_repeater(crop, rules, ctype)

            if crop.get("ready", False):
                # Band 9000; the crop's market value is an in-band tie-break so
                # a ripe melon is picked before a ripe wheat but never outranks
                # the animal-harvest band above it.
                tasks.append(
                    Task(
                        "HARVEST",
                        pos,
                        dr["HARVEST_CROP"]["priority"],
                        value=self._crop_value(rules, ctype, market_stocks),
                        crop=ctype,
                    )
                )

            water_priority = self._water_priority(crop, dr["WATER"], to_die, is_repeater)
            if water_priority is not None:
                tasks.append(Task("WATER", pos, water_priority, crop=ctype))

        # --- Weeds: dig them out --------------------------------------------
        for weed in weeds:
            tasks.append(Task("DIG", tuple(weed), dr["DIG"]["priority"]))

        # --- Planting (within capacity + cash test) + same-day watering ------
        plant_crop = self._choose_plant_crop(rules, step, market_stocks)
        if plant_crop is not None:
            free_slots = max(0, care_capacity - len(crops))
            for tile in empty_tiles[:free_slots]:
                pos = tuple(tile)
                tasks.append(Task("PLANT", pos, dr["PLANT"]["priority"], crop=plant_crop))
                # A newly planted seed already carries one missed-water mark, so
                # it is effectively "about to die": pair a dying-priority water
                # task for the same tile. assign_tasks defers it until the plant
                # actually lands (see below) so no one waters bare ground.
                tasks.append(
                    Task(
                        "WATER",
                        pos,
                        dr["WATER"]["priority_dying"],
                        crop=plant_crop,
                        same_day=True,
                    )
                )

        # --- Animal chores are skipped entirely while animals are disabled ---
        # (Prevention, not post-hoc filtering -- see design doc guardrails.)
        # When enabled, _animal_tasks walks each animal's lifecycle stage and
        # emits HARVEST_ANIMAL (9100), FEED (8800), CARE (4000), plus the setup
        # BUILD_* / PLACE work.
        if animals_on:
            tasks.extend(self._animal_tasks(obs, rules))

        return tasks

    def _animal_tasks(self, obs, rules):
        """Build the animal-lifecycle tasks for this turn.

        Driven entirely by ``obs["animals"]`` -- one dict per animal slot the
        agent operates, mirroring how crops/weeds/empty_tiles drive crop work.
        Recognised per-animal fields (all optional, sensible defaults):
            type:        animal key into ``animal_params`` (GOOSE/COW/SHEEP).
            home_pos:    the ``(x, y)`` tile of its home (build + act target).
            home_built:  the COOP/PASTURE structure exists.
            owned:       we have bought it (it is in the shed or already placed).
            placed:      it is standing in its home and producing.
            fed_today:   it has been fed this day (miss 2 days -> it escapes).
            ready:       a product (egg/milk/wool) is ready to HARVEST.
            needs_care:  a CARE productivity bonus is available to bank.

        Each animal contributes only the task(s) matching its current stage of
        Build home -> Buy -> Pickup+Place -> daily Feed + Care -> Harvest.
        Buying is a market order (see generate_market_orders); the rest is
        worker work routed through the normal band matching.
        """
        dr = rules["daily_routines"]
        animal_params = rules.get("animal_params", {})
        animals = obs.get("animals") or []
        market_stocks = obs.get("market_stocks") or {}
        tasks = []

        for animal in animals:
            atype = animal.get("type")
            params = animal_params.get(atype)
            if not params or animal.get("home_pos") is None:
                continue
            home = tuple(animal["home_pos"])

            # Stage 1: build the home (COOP for geese, PASTURE for cows/sheep).
            if not animal.get("home_built", False):
                build_kind = self._build_action(params.get("home"))
                if build_kind is not None:
                    tasks.append(Task(build_kind, home, dr["BUILD_HOME"]["priority"], crop=atype))
                continue

            # Stage 2: transport a bought-but-unplaced animal. The worker fetches
            # it from the shed (PICKUP), carries it, and PLACEs it on the home.
            if animal.get("owned", False) and not animal.get("placed", False):
                tasks.append(
                    Task("PLACE", home, dr["PLACE_ANIMAL"]["priority"],
                         crop=atype, item=atype, fetch=atype)
                )
                continue

            if not animal.get("placed", False):
                continue

            # Stage 3: a placed animal earns its keep. Bands order these:
            # harvest (9100) > feed (8800) > care (4000).
            if animal.get("ready", False):
                product = params.get("produces")
                tasks.append(
                    Task(
                        "HARVEST",
                        home,
                        dr["HARVEST_ANIMAL"]["priority"],
                        value=self._crop_value(rules, product, market_stocks),
                        crop=atype,
                    )
                )
            if not animal.get("fed_today", False):
                # Feeding needs one wheat carried from the shed first.
                tasks.append(Task("FEED", home, dr["FEED"]["priority"], crop=atype, fetch="WHEAT"))
            if animal.get("needs_care", False):
                tasks.append(Task("CARE", home, dr["CARE"]["priority"], crop=atype))

        return tasks

    # ------------------------------------------------------------------ #
    # Market order generation (purchases)
    # ------------------------------------------------------------------ #
    def generate_market_orders(self, obs, rules, shed_usage=0):
        """Return the ``BUY_PRODUCT`` market orders to place this turn.

        These are engine ``market`` orders (plain lists), emitted alongside the
        morning HIRE/SELL routine in ``main.py``. Purchasing is disabled during
        endgame liquidation (the caller simply does not call us then), so the
        only concern here is the 100-item shed capacity: bought goods land in
        the shed, so we never order more than would fit. ``shed_usage`` is the
        current shed count; we track a running projection as orders accumulate
        so a batch of buys never overshoots the shed together.

        ``BUY_PRODUCT`` is a single opcode; the item (``"FERTILIZER"``, or an
        animal type in Phase 4 step 2) is an argument, never a separate op.
        """
        policy = rules.get("policy", {})
        projected_usage = shed_usage
        orders = []

        # --- Fertilizer -----------------------------------------------------
        if policy.get("FERTILIZER_ENABLED", False):
            qty = self._fertilizer_buy_qty(obs, rules, projected_usage)
            if qty > 0:
                orders.append(["BUY_PRODUCT", "FERTILIZER", qty])
                projected_usage += qty

        # --- Animals: buy an animal once its home is built and it is not yet
        # owned. A bought animal arrives in the shed, so it takes one shed slot
        # until a worker carries it out to its home -- respect capacity.
        if policy.get("ANIMALS_ENABLED", False):
            shed_size = rules["constants"]["shed_size"]
            animal_params = rules.get("animal_params", {})
            for animal in (obs.get("animals") or []):
                atype = animal.get("type")
                if atype not in animal_params:
                    continue
                if (animal.get("home_built", False)
                        and not animal.get("owned", False)
                        and projected_usage + 1 <= shed_size):
                    orders.append(["BUY_PRODUCT", atype])
                    projected_usage += 1

        return orders

    # ------------------------------------------------------------------ #
    # Worker assignment
    # ------------------------------------------------------------------ #
    def assign_tasks(self, workers, tasks, rules, shed_usage=0, drop_mode=None):
        """Match workers to tasks and return ``{worker_id: action_list}``.

        Order of operations mirrors the handover's per-hour ladder:
            1. DROP pre-emption -- a worker whose carried load plus current shed
               usage exceeds the active threshold is pulled out of band
               assignment entirely and routed to the nearest shed tile.
            2. Same-day watering tasks are deferred while their PLANT is still
               pending this turn (a seed cannot be watered before it exists).
            3. Remaining workers are matched to remaining tasks band-by-band via
               greedy minimum-Manhattan-distance pairing.
            4. Any still-idle worker returns PASS (an empty list).
        """
        mode = drop_mode if drop_mode is not None else self.drop_mode
        threshold = self._drop_threshold(rules, mode)

        actions = {}
        free = []

        # 1) DROP pre-emption ------------------------------------------------
        for worker in workers:
            pos = tuple(worker["pos"])
            carried = worker.get("carried", 0)
            # A worker fetching/transporting a task item (wheat for FEED, an
            # animal for PLACE) must not be diverted to DROP -- that would dump
            # the very item it just picked up. Only loose harvest triggers DROP.
            if (carried > 0 and not worker.get("carrying_item")
                    and (carried + shed_usage) > threshold):
                actions[worker["id"]] = self._shed_action(pos)
            else:
                free.append(worker)

        # 2) Defer same-day watering while its PLANT is still queued ---------
        plant_cells = {t.pos for t in tasks if t.kind == "PLANT"}
        active = [t for t in tasks if not (t.same_day and t.pos in plant_cells)]

        # 3) Greedy Manhattan matching, band by band -------------------------
        active.sort(key=Task.sort_key, reverse=True)
        idx, count = 0, len(active)
        while idx < count and free:
            band_key = active[idx].sort_key()
            band = []
            while idx < count and active[idx].sort_key() == band_key:
                band.append(active[idx])
                idx += 1
            self._match_band(free, band, actions)

        # 4) Idle workers pass ----------------------------------------------
        for worker in free:
            actions.setdefault(worker["id"], [])

        return actions

    def _match_band(self, free, band, actions):
        """Greedily assign the closest (worker, task) pairs within one band.

        Repeatedly picks the globally nearest worker/task pair in the band and
        commits it, so total walking inside the band is kept low. Mutates
        ``free`` (removing assigned workers) and ``actions``.
        """
        band_tasks = list(band)
        while free and band_tasks:
            best = None  # (distance, worker_index, task_index)
            for wi, worker in enumerate(free):
                wpos = tuple(worker["pos"])
                for ti, task in enumerate(band_tasks):
                    dist = manhattan(wpos, task.pos)
                    if best is None or dist < best[0]:
                        best = (dist, wi, ti)
            _, wi, ti = best
            worker = free.pop(wi)
            task = band_tasks.pop(ti)
            actions[worker["id"]] = self._task_action(worker, task)

    # ------------------------------------------------------------------ #
    # Action helpers
    # ------------------------------------------------------------------ #
    def _task_action(self, worker, task):
        """Emit the move-or-work action for ``worker`` assigned to ``task``."""
        pos = tuple(worker["pos"])
        if task.kind == "DROP":
            return self._shed_action(pos)
        # Fetch step: FEED/PLACE need an item carried from the shed first. Route
        # the worker to a shed tile and PICKUP before heading to the work tile.
        if task.fetch and worker.get("carrying_item") != task.fetch:
            if pos in SHED_TILES:
                return ["PICKUP", task.fetch]
            return step_towards(pos, nearest_shed_tile(pos))
        if pos == task.pos:
            return task.action()
        return step_towards(pos, task.pos)

    def _shed_action(self, pos):
        """DROP if standing on a shed tile, else step toward the nearest one."""
        if pos in SHED_TILES:
            return ["DROP"]
        return step_towards(pos, nearest_shed_tile(pos))

    # ------------------------------------------------------------------ #
    # Classification helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _is_repeater(crop, rules, ctype):
        """Whether a crop is a repeater (tomato/strawberry)."""
        flagged = crop.get("repeater")
        if flagged is not None:
            return bool(flagged)
        params = rules.get("crop_params", {}).get(ctype)
        return bool(params) and params.get("type") == "repeater"

    @staticmethod
    def _build_action(home_type):
        """Map an animal's home type to the BUILD verb that raises it.

        GOOSE lives in a COOP (``BUILD_COOP``); COW/SHEEP share a PASTURE
        (``BUILD_PASTURE``). Returns None for an unknown home type so an
        unrecognised animal simply produces no build task.
        """
        return {"COOP": "BUILD_COOP", "PASTURE": "BUILD_PASTURE"}.get(home_type)

    @staticmethod
    def _water_priority(crop, water_bands, to_die, is_repeater):
        """Pick the watering band for a crop, or None if it needs no water.

        A plant one mark short of death always wins the top band, regardless of
        whether it was already watered this period -- that is the whole point of
        the 9500 band. Otherwise a task is only raised when the crop still needs
        water this period (``needs_water``):
            * one-time crop inside its yield-bonus window -> bonus (8000)
            * repeater (tomato / strawberry)              -> ongoing (7000)
            * anything else                               -> normal (5000)
        """
        if crop.get("misses", 0) >= to_die - 1:
            return water_bands["priority_dying"]
        if not crop.get("needs_water", False):
            return None
        if crop.get("in_bonus_window", False):
            return water_bands["priority_bonus"]
        if is_repeater:
            return water_bands["priority_ongoing"]
        return water_bands["priority_normal"]

    @staticmethod
    def _crop_value(rules, ctype, market_stocks):
        """Market value of a crop, used as the HARVEST in-band tie-break.

        Uses the live single-unit market price at the current stock so a crop
        whose market has already crashed is not over-prioritized.
        """
        stock = market_stocks.get(ctype, 0) if market_stocks else 0
        return float(MarketModel.market_price(rules, ctype, stock))

    @staticmethod
    def _fertilizer_buy_qty(obs, rules, shed_usage):
        """How many FERTILIZER units to buy this turn (0 = none).

        Fertilizer is worth buying only when standing crops would actually
        yield more with it (``yield_fertilized > yield_no_fertilizer`` -- true
        for wheat/carrot, not melon) and we do not already hold enough in the
        shed. The order is clamped to the free shed slots so a purchase can
        never push the shed past ``shed_size``: bought fertilizer occupies a
        shed slot until a worker picks it up to apply.
        """
        crops = obs.get("crops") or []
        shed = obs.get("shed") or {}
        shed_size = rules["constants"]["shed_size"]
        crop_params = rules.get("crop_params", {})

        beneficial = 0
        for crop in crops:
            params = crop_params.get(crop.get("type"))
            if not params:
                continue
            if params.get("yield_fertilized", 0) > params.get("yield_no_fertilizer", 0):
                beneficial += 1
        if beneficial == 0:
            return 0

        have = shed.get("FERTILIZER", 0) or 0
        need = max(0, beneficial - have)
        capacity_left = max(0, shed_size - shed_usage)
        return min(need, capacity_left)

    def _choose_plant_crop(self, rules, step, market_stocks):
        """Best-EV crop that can still be harvested and sold before the season
        ends, or None if nothing passes the cash test (then we plant nothing)."""
        ranked = MarketModel.crop_values(rules, market_stocks or {})
        for ctype in ranked:
            if self._can_finish(ctype, rules, step):
                return ctype
        return None

    @staticmethod
    def _can_finish(ctype, rules, step):
        """Planting cash test: can this crop mature (and leave a day to sell)
        before turn 720? Cutoff day = season_days - grow_days - 1, which
        reproduces the handover's carrot 26 / wheat 25 / melon 19 cutoffs."""
        params = rules.get("crop_params", {}).get(ctype)
        if not params:
            return False
        season_days = rules["constants"]["season_length"] // 24
        day = step // 24
        if params.get("type") == "repeater":
            grow_days = params.get("first_fruit", season_days)
        else:
            grow_days = params.get("full_harvest", season_days)
        return day <= season_days - grow_days - 1

    def _drop_threshold(self, rules, mode):
        """Item count above which a carrying worker is sent to the shed.

        normal   -> policy.drop_pressure of the shed (default 0.8 -> 80 items)
        overflow -> half the shed (tighten when a spill is imminent)
        endgame  -> 0 (every carried item must be banked before the clock ends)
        """
        shed_size = rules["constants"]["shed_size"]
        if mode == "endgame":
            return 0
        if mode == "overflow":
            return 0.5 * shed_size
        return rules["policy"]["drop_pressure"] * shed_size
